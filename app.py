# SECTION 1: IMPORTS
import os
import json
import re
import hashlib
import requests
import textwrap
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from PIL import Image

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from ddgs import DDGS
from tenacity import retry, stop_after_attempt, wait_exponential

import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from econml.dml import LinearDML, CausalForestDML
from econml.score import RScorer


# SECTION 2: API KEY & MODEL CONFIG
API_KEY_FILE = ".api_key"
CACHE_DIR = "./cache/"
os.makedirs(CACHE_DIR, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "CognetAdvisor.png")
if not os.path.exists(LOGO_PATH):
    LOGO_PATH = None

DEFAULT_MODEL = "gpt-6-luna"
FALLBACK_MODEL = "gpt-3.5-turbo"
MODEL_NAME = os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)


def load_api_key():
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, "r") as f:
            return f.read().strip()
    return None


def save_api_key(key: str):
    with open(API_KEY_FILE, "w") as f:
        f.write(key.strip())


def clear_api_key():
    if os.path.exists(API_KEY_FILE):
        os.remove(API_KEY_FILE)
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]


def get_model_name():
    try:
        if "OPENAI_MODEL" in st.secrets:
            return st.secrets["OPENAI_MODEL"]
    except Exception:
        pass
    return MODEL_NAME


def build_llm():
    primary = get_model_name()
    try:
        is_reasoning_model = primary.startswith("gpt-6") or primary.startswith("o1") or primary.startswith("o3")
        if is_reasoning_model:
            client = ChatOpenAI(model=primary) 
        else:
            client = ChatOpenAI(model=primary, temperature=0.2)
        client.invoke("ping")
        return client, primary
    except Exception as e:
        try:
            st.warning(
                f"⚠ Model `{primary}` unavailable ({str(e)[:120]}). "
                f"Falling back to `{FALLBACK_MODEL}`. "
                f"Upgrade your API tier or upgrade `openai` / `langchain-openai` "
                f"to enable `{primary}`."
            )
        except Exception:
            pass
        return ChatOpenAI(model=FALLBACK_MODEL, temperature=0.2), FALLBACK_MODEL


def check_password() -> bool:
    try:
        stored = st.secrets["APP_PASSWORD"] if "APP_PASSWORD" in st.secrets else None
    except Exception:
        stored = None
    if not stored:
        stored = os.environ.get("APP_PASSWORD", None)
    if not stored:
        return True

    def enter():
        if st.session_state.get("password", "") == stored:
            st.session_state["password_correct"] = True
            if "password" in st.session_state:
                del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("🔒 Enter Access Password", type="password",
                      on_change=enter, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("🔒 Enter Access Password", type="password",
                      on_change=enter, key="password")
        st.error("😕 Password incorrect")
        return False
    return True


# SECTION 2.5: SOURCE DETECTION
def get_country_domains(country: str) -> List[str]:
    m = {
        "Pakistan": ['.pk', '.edu.pk'], "USA": ['.us', '.edu'], "UK": ['.uk', '.ac.uk'],
        "Canada": ['.ca'], "Australia": ['.au', '.edu.au'], "India": ['.in', '.ac.in'],
        "Germany": ['.de'], "France": ['.fr'], "China": ['.cn', '.edu.cn'],
        "Japan": ['.jp', '.ac.jp'], "South Korea": ['.kr', '.ac.kr'],
        "Singapore": ['.sg', '.edu.sg'], "Malaysia": ['.my', '.edu.my'],
        "UAE": ['.ae', '.ac.ae'], "Saudi Arabia": ['.sa', '.edu.sa'],
        "Egypt": ['.eg', '.edu.eg'], "South Africa": ['.za', '.ac.za'],
        "Brazil": ['.br', '.edu.br'], "Mexico": ['.mx', '.edu.mx'],
        "Italy": ['.it'], "Spain": ['.es'], "Netherlands": ['.nl'],
        "Sweden": ['.se'], "Norway": ['.no'], "Russia": ['.ru'], "Other": []
    }
    return m.get(country, [])


def clean_source_url(url: str) -> str:
    if not url:
        return ""
    for p in ["Scraped from ", "Scraped from: ", "Scraped from - "]:
        if url.startswith(p):
            return url[len(p):].strip()
    return url.strip()


def detect_course_source(source_url: str, student_country: str,
                          student_university: str = None) -> Tuple[str, str]:
    if not source_url or source_url in ['Unknown',
                                         'Semantic estimation (no content found)',
                                         'Fallback']:
        return "semantic", "Semantic Estimation"
    sl = source_url.lower()
    cds = get_country_domains(student_country)
    uni_match = False
    if student_university:
        for part in student_university.lower().replace(' ', '').split(','):
            if part.strip() and part.strip() in sl:
                uni_match = True
                break
    if uni_match or any(d in sl for d in cds):
        return "regional", f"Regional ({student_country})"
    
    # FIX: Detect any valid URL as International if it's not the home country
    if sl.startswith("http"):
        return "international", "International (Foreign URL)"
        
    return "semantic", "Semantic Estimation"


def get_course_content_comparison(course_name: str, source_type: str, features: Dict) -> Dict:
    c = {
        "regional": {
            "format": "Detailed PDF syllabus",
            "structure": "12-16 weeks of weekly topics",
            "credit_hours": f"{features.get('credit_hours', '3+1')} (theory + lab)"
                if features.get('credit_hours', 'Unknown') != 'Unknown' else "Not specified",
            "lab_component": "1 credit hour lab" if str(features.get('credit_hours', '')).endswith('+1')
                else "No lab specified",
            "assessment": "Midterm, Final, Assignments, Lab Work",
            "textbook": "Specific local textbook references",
            "learning_style": "Structured, sequential, in-depth"
        },
        "international": {
            "format": "Brief HTML description",
            "structure": "4-6 modules (self-paced)",
            "credit_hours": "Not available",
            "lab_component": "Rarely specified",
            "assessment": "Quizzes, Projects",
            "textbook": "General recommendations",
            "learning_style": "Flexible, self-paced, broad overview"
        },
        "semantic": {
            "format": "No syllabus found",
            "structure": "Estimated from course name",
            "credit_hours": "Estimated", "lab_component": "Estimated",
            "assessment": "Estimated", "textbook": "Estimated",
            "learning_style": "Estimated"
        }
    }
    return c.get(source_type, c["semantic"])


def get_learning_resources(course_name: str) -> Dict[str, List[str]]:
    m = {
        "Computer Graphics": {
            "Coursera": ["Computer Graphics - UC San Diego", "Interactive Computer Graphics - Stanford"],
            "Udemy": ["Learn Computer Graphics with OpenGL", "Game Development & Computer Graphics"],
            "MIT OpenCourseWare": ["Computer Graphics (6.837)", "Introduction to Computer Graphics (6.856)"],
            "YouTube": ["The Cherno", "Two Minute Papers"]
        },
        "Artificial Intelligence": {
            "Coursera": ["AI for Everyone - Andrew Ng", "Machine Learning - Stanford"],
            "edX": ["Artificial Intelligence - Columbia", "Machine Learning - MIT"],
            "Udacity": ["Intro to Artificial Intelligence", "Deep Learning Nanodegree"],
            "Kaggle": ["Machine Learning Courses", "Data Science Courses"]
        },
        "Data Science": {
            "Coursera": ["Data Science - Johns Hopkins", "Python for Data Science - IBM"],
            "DataCamp": ["Data Scientist with Python", "Machine Learning Scientist"],
            "Kaggle": ["Intro to Data Science", "Machine Learning Courses"],
            "edX": ["Data Science - Harvard", "Analytics - MIT"]
        },
        "Data Structures & Algorithm": {
            "Coursera": ["Algorithms, Part I - Princeton", "Data Structures and Algorithms - UC San Diego"],
            "edX": ["Data Structures - MIT", "Algorithm Design and Analysis - Penn"],
            "Udemy": ["Master the Coding Interview: Data Structures + Algorithms"],
            "GeeksforGeeks": ["DSA Self Paced"]
        },
        "Theory of Automata": {
            "Coursera": ["Automata Theory - Stanford", "Introduction to Automata - IIT"],
            "edX": ["Automata Theory - MIT", "Formal Languages and Automata - TU Delft"],
            "Neso Academy": ["Theory of Computation Playlist"],
            "YouTube": ["Computer Science Theory - MIT OpenCourseWare"]
        },
        "Introduction to Data Science": {
            "Coursera": ["Data Science Specialization - Johns Hopkins", "Introduction to Data Science - IBM"],
            "edX": ["Data Science Essentials - Microsoft", "Introduction to Data Science - Harvard"],
            "DataCamp": ["Data Scientist with Python", "Data Science Fundamentals"],
            "Kaggle": ["Intro to Data Science"]
        },
        "Software Engineering": {
            "Coursera": ["Software Engineering - University of Alberta", "Software Development Lifecycle"],
            "edX": ["Software Engineering - MIT", "Agile Development - University of Virginia"],
            "Udemy": ["Software Engineering 101", "Agile & Scrum"]
        },
        "Computer Networks": {
            "Coursera": ["Computer Networks - University of Washington", "Network Security - Stanford"],
            "Udemy": ["The Complete Networking Fundamentals Course", "Cisco CCNA"],
            "Cisco Networking Academy": ["CCNA Introduction to Networks", "CCNA Switching"],
            "edX": ["Computer Networking - Georgia Tech", "Network Security - MIT"]
        }
    }
    return m.get(course_name, {
        "Coursera": [f"Search for courses related to {course_name}"],
        "Udemy": [f"Search for courses related to {course_name}"],
        "edX": [f"Search for courses related to {course_name}"]
    })


def get_learning_resources_with_urls(course_name: str) -> Dict[str, List[Tuple[str, str]]]:
    m = {
        "Computer Graphics": {
            "Coursera": [
                ("Computer Graphics - UC San Diego", "https://www.coursera.org/learn/computer-graphics"),
                ("Interactive Computer Graphics - Stanford", "https://www.coursera.org/learn/interactive-computer-graphics")
            ],
            "Udemy": [
                ("Learn Computer Graphics with OpenGL", "https://www.udemy.com/course/learn-computer-graphics-with-opengl/"),
                ("Game Development & Computer Graphics", "https://www.udemy.com/course/game-development-computer-graphics/")
            ],
            "MIT OpenCourseWare": [
                ("Computer Graphics (6.837)", "https://ocw.mit.edu/courses/6-837-computer-graphics-fall-2012/"),
                ("Introduction to Computer Graphics (6.856)", "https://ocw.mit.edu/courses/6-856-introduction-to-computer-graphics-fall-2015/")
            ],
            "YouTube": [
                ("The Cherno", "https://www.youtube.com/user/TheCherno"),
                ("Two Minute Papers", "https://www.youtube.com/user/keeroyz")
            ]
        },
        "Artificial Intelligence": {
            "Coursera": [
                ("AI for Everyone - Andrew Ng", "https://www.coursera.org/learn/ai-for-everyone"),
                ("Machine Learning - Stanford", "https://www.coursera.org/learn/machine-learning")
            ],
            "edX": [
                ("Artificial Intelligence - Columbia", "https://www.edx.org/course/artificial-intelligence-ai"),
                ("Machine Learning - MIT", "https://www.edx.org/course/machine-learning")
            ],
            "Udacity": [
                ("Intro to Artificial Intelligence", "https://www.udacity.com/course/intro-to-artificial-intelligence--cs271"),
                ("Deep Learning Nanodegree", "https://www.udacity.com/course/deep-learning-nanodegree--nd101")
            ],
            "Kaggle": [
                ("Machine Learning Courses", "https://www.kaggle.com/learn"),
                ("Data Science Courses", "https://www.kaggle.com/learn/data-science")
            ]
        },
        "Data Science": {
            "Coursera": [
                ("Data Science - Johns Hopkins", "https://www.coursera.org/specializations/jhu-data-science"),
                ("Python for Data Science - IBM", "https://www.coursera.org/learn/python-for-applied-data-science-ai")
            ],
            "DataCamp": [
                ("Data Scientist with Python", "https://www.datacamp.com/tracks/data-scientist-with-python"),
                ("Machine Learning Scientist", "https://www.datacamp.com/tracks/machine-learning-scientist-with-python")
            ],
            "Kaggle": [
                ("Intro to Data Science", "https://www.kaggle.com/learn/intro-to-data-science"),
                ("Machine Learning Courses", "https://www.kaggle.com/learn")
            ],
            "edX": [
                ("Data Science - Harvard", "https://www.edx.org/professional-certificate/harvardx-data-science"),
                ("Analytics - MIT", "https://www.edx.org/course/data-science-and-analytics")
            ]
        },
        "Data Structures & Algorithm": {
            "Coursera": [
                ("Algorithms, Part I - Princeton", "https://www.coursera.org/learn/algorithms-part1"),
                ("Data Structures and Algorithms - UC San Diego", "https://www.coursera.org/specializations/data-structures-algorithms")
            ],
            "edX": [
                ("Data Structures - MIT", "https://www.edx.org/course/data-structures"),
                ("Algorithm Design and Analysis - Penn", "https://www.edx.org/course/algorithm-design-and-analysis")
            ],
            "Udemy": [
                ("Master the Coding Interview", "https://www.udemy.com/course/master-the-coding-interview-data-structures-algorithms/")
            ],
            "GeeksforGeeks": [
                ("DSA Self Paced", "https://www.geeksforgeeks.org/courses/dsa-self-paced")
            ]
        },
        "Theory of Automata": {
            "Coursera": [
                ("Automata Theory - Stanford", "https://www.coursera.org/learn/automata"),
                ("Introduction to Automata - IIT", "https://www.coursera.org/learn/automata-theory")
            ],
            "edX": [
                ("Automata Theory - MIT", "https://www.edx.org/course/automata-theory"),
                ("Formal Languages and Automata - TU Delft", "https://www.edx.org/course/formal-languages-automata")
            ],
            "Neso Academy": [
                ("Theory of Computation Playlist", "https://www.youtube.com/playlist?list=PLBlnK6fEyqRgp46KUv4ZY69yXmpwKOIev")
            ],
            "YouTube": [
                ("MIT OpenCourseWare", "https://www.youtube.com/playlist?list=PLUl4u3cNGP60_UYpFj6C3tN0N_Tn2dAL6")
            ]
        },
        "Introduction to Data Science": {
            "Coursera": [
                ("Data Science Specialization - Johns Hopkins", "https://www.coursera.org/specializations/jhu-data-science"),
                ("Introduction to Data Science - IBM", "https://www.coursera.org/learn/what-is-data-science")
            ],
            "edX": [
                ("Data Science Essentials - Microsoft", "https://www.edx.org/course/data-science-essentials"),
                ("Introduction to Data Science - Harvard", "https://www.edx.org/course/data-science-introduction")
            ],
            "DataCamp": [
                ("Data Scientist with Python", "https://www.datacamp.com/tracks/data-scientist-with-python"),
                ("Data Science Fundamentals", "https://www.datacamp.com/courses/intro-to-data-science")
            ],
            "Kaggle": [
                ("Intro to Data Science", "https://www.kaggle.com/learn/intro-to-data-science")
            ]
        },
        "Computer Networks": {
            "Coursera": [
                ("Computer Networks - UW", "https://www.coursera.org/learn/computer-networking"),
                ("Network Security - Stanford", "https://www.coursera.org/learn/network-security")
            ],
            "Udemy": [
                ("Complete Networking Fundamentals", "https://www.udemy.com/course/complete-networking-fundamentals-course/"),
                ("Cisco CCNA", "https://www.udemy.com/course/ccna-200-301/")
            ],
            "Cisco Networking Academy": [
                ("CCNA Introduction to Networks", "https://www.netacad.com/courses/ccna-introduction-networks"),
                ("CCNA Switching", "https://www.netacad.com/courses/ccna-switching-routing-wireless-essentials")
            ],
            "edX": [
                ("Computer Networking - Georgia Tech", "https://www.edx.org/course/computer-networking"),
                ("Network Security - MIT", "https://www.edx.org/course/network-security")
            ]
        }
    }
    if course_name not in m:
        return {
            "Coursera": [(f"Search for {course_name}", f"https://www.coursera.org/search?query={course_name.replace(' ', '%20')}")],
            "Udemy": [(f"Search for {course_name}", f"https://www.udemy.com/courses/search/?q={course_name.replace(' ', '+')}")],
            "edX": [(f"Search for {course_name}", f"https://www.edx.org/search?q={course_name.replace(' ', '+')}")]
        }
    return m[course_name]


# SECTION 3: FSLSM QUESTION BANK
FSLSM_QUESTIONS = [
    ("I understand something better after I (a) try it out. (b) think it through.", "ACT_REF", "a"),
    ("I would rather be considered (a) realistic. (b) innovative.", "SEN_INT", "a"),
    ("When I think about what I did yesterday, I am most likely to get (a) a picture. (b) words.", "VIS_VER", "a"),
    ("I tend to (a) understand details of a subject but may be fuzzy about its overall structure. (b) understand the overall structure but may be fuzzy about details.", "SEQ_GLO", "a"),
    ("When I am learning something new, it helps me to (a) talk about it. (b) think about it.", "ACT_REF", "a"),
    ("If I were a teacher, I would rather teach a course (a) that deals with facts and real life situations. (b) that deals with ideas and theories.", "SEN_INT", "a"),
    ("I prefer to get new information in (a) pictures, diagrams, graphs, or maps. (b) written directions or verbal information.", "VIS_VER", "a"),
    ("Once I understand (a) all the parts, I understand the whole thing. (b) the whole thing, I see how the parts fit.", "SEQ_GLO", "a"),
    ("In a study group working on difficult material, I am more likely to (a) jump in and contribute ideas. (b) sit back and listen.", "ACT_REF", "a"),
    ("I find it easier (a) to learn facts. (b) to learn concepts.", "SEN_INT", "a"),
    ("In a book with lots of pictures and charts, I am likely to (a) look over the pictures and charts carefully. (b) focus on the written text.", "VIS_VER", "a"),
    ("When I solve math problems (a) I usually work my way to the solutions one step at a time. (b) I often just see the solutions but then have to struggle to figure out the steps to get to them.", "SEQ_GLO", "a"),
    ("In classes I have taken (a) I have usually gotten to know many of the students. (b) I have rarely gotten to know many of the students.", "ACT_REF", "a"),
    ("In reading nonfiction, I prefer (a) something that teaches me new facts or tells me how to do something. (b) something that gives me new ideas to think about.", "SEN_INT", "a"),
    ("I like teachers (a) who put a lot of diagrams on the board. (b) who spend a lot of time explaining.", "VIS_VER", "a"),
    ("When I'm analyzing a story or a novel (a) I think of the incidents and try to put them together to figure out the themes. (b) I just know what the themes are when I finish reading and then I have to go back and find the incidents that demonstrate them.", "SEQ_GLO", "a"),
    ("When I start a homework problem, I am more likely to (a) start working on the solution immediately. (b) try to fully understand the problem first.", "ACT_REF", "a"),
    ("I prefer the idea of (a) certainty. (b) theory.", "SEN_INT", "a"),
    ("I remember best (a) what I see. (b) what I hear.", "VIS_VER", "a"),
    ("It is more important to me that an instructor (a) lay out the material in clear sequential steps. (b) give me an overall picture and relate the material to other subjects.", "SEQ_GLO", "a"),
    ("I prefer to study (a) in a study group. (b) alone.", "ACT_REF", "a"),
    ("I am more likely to be considered (a) careful about the details of my work. (b) creative about how to do my work.", "SEN_INT", "a"),
    ("When I get directions to a new place, I prefer (a) a map. (b) written instructions.", "VIS_VER", "a"),
    ("I learn (a) at a fairly regular pace. If I study hard, I'll 'get it.' (b) in fits and starts. I'll be totally confused and then suddenly it all 'clicks.'", "SEQ_GLO", "a"),
    ("I would rather first (a) try things out. (b) think about how I'm going to do it.", "ACT_REF", "a"),
    ("When I am reading for enjoyment, I like writers to (a) clearly say what they mean. (b) say things in creative, interesting ways.", "SEN_INT", "a"),
    ("When I see a diagram or sketch in class, I am most likely to remember (a) the picture. (b) what the instructor said about it.", "VIS_VER", "a"),
    ("When considering a body of information, I am more likely to (a) focus on details and miss the big picture. (b) try to understand the big picture before getting into the details.", "SEQ_GLO", "a"),
    ("I more easily remember (a) something I have done. (b) something I have thought a lot about.", "ACT_REF", "a"),
    ("When I have to perform a task, I prefer to (a) master one way of doing it. (b) come up with new ways of doing it.", "SEN_INT", "a"),
    ("When someone is showing me data, I prefer (a) charts or graphs. (b) text summarizing the results.", "VIS_VER", "a"),
    ("When writing a paper, I am more likely to (a) work on the beginning and progress forward. (b) work on different parts and then order them.", "SEQ_GLO", "a"),
    ("When I have to work on a group project, I first want to (a) have 'group brainstorming'. (b) brainstorm individually and then come together.", "ACT_REF", "a"),
    ("I consider it higher praise to call someone (a) sensible. (b) imaginative.", "SEN_INT", "a"),
    ("When I meet people at a party, I am more likely to remember (a) what they looked like. (b) what they said about themselves.", "VIS_VER", "a"),
    ("When I am learning a new subject, I prefer to (a) stay focused on that subject. (b) try to make connections between that subject and related subjects.", "SEQ_GLO", "a"),
    ("I am more likely to be considered (a) outgoing. (b) reserved.", "ACT_REF", "a"),
    ("I prefer courses that emphasize (a) concrete material. (b) abstract material.", "SEN_INT", "a"),
    ("For entertainment, I would rather (a) watch television. (b) read a book.", "VIS_VER", "a"),
    ("Some teachers start their lectures with an outline. Such outlines are (a) somewhat helpful. (b) very helpful.", "SEQ_GLO", "a"),
    ("The idea of doing homework in groups with one grade (a) appeals to me. (b) does not appeal to me.", "ACT_REF", "a"),
    ("When I am doing long calculations, (a) I tend to repeat all my steps and check my work carefully. (b) I find checking my work tiresome.", "SEQ_GLO", "a"),
    ("I tend to picture places I have been (a) easily and fairly accurately. (b) with difficulty and without much detail.", "VIS_VER", "a"),
    ("When solving problems in a group, I would be more likely to (a) think of the steps in the solution process. (b) think of possible consequences or applications of the solution.", "SEQ_GLO", "a"),
]


def extract_options(q_text):
    match = re.search(r'\(a\)\s*(.+?)\s*\(b\)\s*(.+)', q_text)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "a", "b"


# SECTION 4: UTILITY
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def safe_llm_call(chain, input_data):
    return chain.invoke(input_data)


def build_rag_from_pdf(pdf_path: str, embeddings) -> Optional[Chroma]:
    file_hash = hashlib.md5(open(pdf_path, "rb").read()).hexdigest()[:8]
    persist_dir = f"./chroma_{file_hash}"
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        vs = Chroma.from_documents(documents=chunks, embedding=embeddings,
                                    persist_directory=persist_dir)
        vs.persist()
        return vs
    except Exception:
        return None


# SECTION 5: RAG PREPARER
class BackgroundRAGPreparer:
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.is_ready = False
        self.course_features = {}
        self.error = None
        self._last_success_url = None
        self.log: List[str] = []

    def _log(self, msg):
        self.log.append(msg)
        print(f"[RAG] {msg}")

    def _validate_pdf(self, content: bytes) -> bool:
        return content.startswith(b'%PDF') and len(content) >= 5120

    def _clean_url(self, url):
        import urllib.parse
        try:
            p = urllib.parse.urlparse(url)
            path = urllib.parse.unquote(p.path)
            encoded = urllib.parse.quote(path, safe="/,&+()'")
            return urllib.parse.urlunparse((p.scheme, p.netloc, encoded, p.params,
                                             p.query, p.fragment))
        except Exception:
            return url

    def _try_download_variants(self, base_url, timeout=15):
        import urllib.parse
        variants = [
            base_url, base_url.replace('+', '%20'), base_url.replace('+', '_'),
            base_url.replace('%20', '+'), base_url.replace(' ', '%20'),
            base_url.replace(' ', '+'), base_url.replace('%20', '_'),
            base_url.replace(' ', '+').replace('+', ''), base_url.replace('%20', '')
        ]
        try:
            dec = urllib.parse.unquote(base_url)
            pd = urllib.parse.urlparse(dec)
            variants.append(urllib.parse.urlunparse((pd.scheme, pd.netloc,
                                                      pd.path.replace(' ', '%20'),
                                                      pd.params, pd.query, pd.fragment)))
            variants.append(urllib.parse.urlunparse((pd.scheme, pd.netloc,
                                                      pd.path.replace('&', '%26'),
                                                      pd.params, pd.query, pd.fragment)))
        except Exception:
            pass
        variants = list(dict.fromkeys(variants))
        for u in variants:
            try:
                r = requests.get(u, timeout=timeout, verify=False, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/pdf,text/html,*/*;q=0.9'
                })
                ct = r.headers.get('Content-Type', '').lower()
                if r.status_code == 200:
                    if 'text/html' in ct or 'text/plain' in ct:
                        continue
                    content = r.content
                    if (content.startswith(b'<!DOCTYPE') or content.startswith(b'<html')
                            or b'<head' in content[:200]):
                        continue
                    if self._validate_pdf(content):
                        fname = hashlib.md5(u.encode()).hexdigest()[:8] + '.pdf'
                        return (content, fname, u)
            except Exception:
                continue
        return None

    def _download_pdf(self, url, timeout=15):
        cleaned = self._clean_url(url)
        path = os.path.join(CACHE_DIR, hashlib.md5(url.encode()).hexdigest()[:8] + '.pdf')
        if os.path.exists(path) and os.path.getsize(path) > 5120:
            with open(path, 'rb') as f:
                if f.read(4) == b'%PDF':
                    return path
            try:
                os.remove(path)
            except Exception:
                pass
        result = (self._try_download_variants(cleaned, timeout)
                  or self._try_download_variants(url, timeout))
        if result:
            content, fname, success_url = result
            path = os.path.join(CACHE_DIR, fname)
            with open(path, 'wb') as f:
                f.write(content)
            self._last_success_url = success_url
            return path
        return None

    def _scrape_course_page(self, url, timeout=15):
        from bs4 import BeautifulSoup
        try:
            r = requests.get(url, timeout=timeout, verify=False, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                for el in soup(['script', 'style', 'meta', 'link', 'noscript']):
                    el.decompose()
                text = ' '.join(soup.get_text(separator=' ').split())
                if len(text) > 50 and any(kw in text.lower() for kw in
                                          ['syllabus', 'credit', 'outline', 'topic',
                                           'course', 'week', 'module']):
                    return text
        except Exception:
            pass
        return None

    def _search_university_domain(self, course, degree, university, country):
        try:
            uni_parts = [p.lower() for p in university.split() if len(p) > 3]
            base = university.lower().replace(' ', '').replace('&', '').replace('.', '')
            domains = [f"{base}.edu.pk", f"{base}.pk", f"{base}.edu"]
            words = university.split()
            if len(words) >= 2:
                acr = ''.join([w[0].lower() for w in words if w[0].isalpha()])
                domains += [f"{acr}.edu.pk", f"{acr}.pk", f"{acr}.edu"]
            with DDGS() as ddgs:
                queries = [
                    f"{course} {university} {degree} syllabus",
                    f"{course} {university} course outline",
                    f"{course} {university} syllabus filetype:pdf",
                    f"{course} {university} {degree} curriculum",
                    f"{course} {university} academic courses",
                ]
                if domains:
                    queries += [f"{course} syllabus site:{domains[0]}",
                                f"{course} course filetype:pdf site:{domains[0]}"]
                for q in queries:
                    try:
                        results = list(ddgs.text(q, max_results=5))
                    except Exception:
                        continue
                    for r in results:
                        url = (r.get('href') or '').strip()
                        if not url:
                            continue
                        ul = url.lower()
                        valid = any(d and d in ul for d in domains) or any(p in ul for p in uni_parts)
                        if not valid:
                            continue
                        cd = ".pk" if country.lower() == "pakistan" else ".edu" if country.lower() == "usa" else ""
                        if cd and cd not in ul:
                            continue
                        return (url, 'pdf' if '.pdf' in ul else 'html')
        except Exception:
            pass
        return None

    def _search_course_outline(self, course, degree, country, university=None):
        try:
            with DDGS() as ddgs:
                cc = country.lower()
                queries = []
                if university:
                    queries.append(f"{course} {university} {degree} syllabus filetype:pdf")
                queries += [
                    f"{course} {degree} syllabus site:.{cc[:2]} filetype:pdf",
                    f"{course} course outline {country} filetype:pdf",
                    f"{course} {degree} syllabus filetype:pdf",
                    f"{course} syllabus filetype:pdf",
                ]
                if cc == "pakistan":
                    queries.append(f"{course} {degree} syllabus Pakistan filetype:pdf")
                for q in queries:
                    try:
                        results = list(ddgs.text(q, max_results=5))
                    except Exception:
                        continue
                    for r in results:
                        url = (r.get('href') or '').strip()
                        if url and '.pdf' in url.lower():
                            return url
        except Exception:
            pass
        return None

    def _extract_features_from_context(self, course, context):
        try:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="Rate this course on Visual, Practical, Structured (0-1). Output strict JSON: C_vis, C_prac, C_struct."),
                HumanMessage(content=f"Course: {course}\nContext: {context[:2000]}")
            ])
            res = safe_llm_call(prompt | self.llm | StrOutputParser(), {})
            match = re.search(r'\{.*\}', res, re.DOTALL)
            f = json.loads(match.group()) if match else {"C_vis": 0.5, "C_prac": 0.5, "C_struct": 0.5}
            f["C_vis"] = max(0.2, min(1.0, f.get("C_vis", 0.5)))
            f["C_prac"] = max(0.1, min(1.0, f.get("C_prac", 0.5)))
            f["C_struct"] = max(0.1, min(1.0, f.get("C_struct", 0.5)))
            if f["C_vis"] >= 0.95 and f["C_prac"] >= 0.95 and f["C_struct"] >= 0.95:
                f["C_vis"], f["C_prac"], f["C_struct"] = 0.9, 0.8, 0.8

            cp = ChatPromptTemplate.from_messages([
                SystemMessage(content="Extract ONLY credit hours from context. Output only the pattern like '3+1'. If not found, output 'Unknown'."),
                HumanMessage(content=f"Context: {context[:1500]}")
            ])
            raw = safe_llm_call(cp | self.llm | StrOutputParser(), {}).strip()
            ch = "Unknown"
            for pat in [r'(\d+)\s*\+\s*(\d+)', r'Credit Hours?\s*:\s*(\d+)',
                        r'(\d+)\s*credits', r'(\d+)\s*Cr\.\s*Hrs\.', r'(\d+)\s*CH']:
                m = re.search(pat, raw, re.IGNORECASE)
                if m:
                    ch = f"{m.group(1)}+{m.group(2)}" if '+' in pat else f"{m.group(1)}+0"
                    break
            if ch == "Unknown":
                lab_kw = ["lab", "practical", "programming", "graphics", "networks",
                          "compiler", "database", "web", "software", "game", "hci",
                          "human computer", "interaction"]
                ch = "3+1" if any(k in course.lower() for k in lab_kw) else "3+0"
            f["credit_hours"] = ch
            return f
        except Exception:
            return {"C_vis": 0.5, "C_prac": 0.5, "C_struct": 0.5, "credit_hours": "3+0"}

    def _apply_semantic_fallback(self, course):
        try:
            p = ChatPromptTemplate.from_messages([
                SystemMessage(content="Estimate Visual, Practical, Structured scores for this CS course. Output JSON: C_vis, C_prac, C_struct."),
                HumanMessage(content=f"Course: {course}")
            ])
            res = safe_llm_call(p | self.llm | StrOutputParser(), {})
            m = re.search(r'\{.*\}', res, re.DOTALL)
            f = json.loads(m.group()) if m else {"C_vis": 0.5, "C_prac": 0.5, "C_struct": 0.5}
            f["C_vis"] = max(0.2, min(1.0, f.get("C_vis", 0.5)))
            f["C_prac"] = max(0.1, min(1.0, f.get("C_prac", 0.5)))
            f["C_struct"] = max(0.1, min(1.0, f.get("C_struct", 0.5)))
            lab_kw = ["lab", "practical", "programming", "graphics", "networks",
                      "compiler", "database", "web", "software", "game", "hci",
                      "human computer", "interaction"]
            f["credit_hours"] = "3+1" if any(k in course.lower() for k in lab_kw) else "3+0"
            return f
        except Exception:
            return {"C_vis": 0.5, "C_prac": 0.5, "C_struct": 0.5, "credit_hours": "3+0"}

    def _ensure_distinct_features(self, fd):
        if len(fd) <= 1:
            return fd
        first = next(iter(fd.values()))
        if not all(abs(fd[c]["C_vis"] - first["C_vis"]) < 1e-9 and
                   abs(fd[c]["C_prac"] - first["C_prac"]) < 1e-9 and
                   abs(fd[c]["C_struct"] - first["C_struct"]) < 1e-9 for c in fd):
            return fd
        for c in fd:
            new = self._apply_semantic_fallback(c)
            new["source_url"] = fd[c].get("source_url", "Semantic estimation (no content found)")
            new["credit_hours"] = fd[c].get("credit_hours", new["credit_hours"])
            fd[c] = new
        return fd

    def _process_html_url(self, course, url):
        scraped = self._scrape_course_page(url)
        if scraped:
            f = self._extract_features_from_context(course, scraped)
            f["source_url"] = url
            return f
        return None

    def _process_url(self, course, url, url_type):
        if not url:
            return None
        is_pdf = (url_type == 'pdf') or url.lower().split('?')[0].endswith('.pdf')
        if is_pdf:
            path = self._download_pdf(url)
            if path:
                with open(path, 'rb') as f:
                    header = f.read(8)
                if not header.startswith(b'%PDF'):
                    self._log(f"{course}: 'PDF' was actually HTML, scraping...")
                    return self._process_html_url(course, url)
                vs = build_rag_from_pdf(path, self.embeddings)
                if vs:
                    docs = vs.similarity_search(
                        f"Course {course} description objectives outline credit hours", k=5)
                    ctx = "\n".join([d.page_content for d in docs])
                    f = self._extract_features_from_context(course, ctx)
                    f["source_url"] = self._last_success_url or url
                    return f
            self._log(f"{course}: PDF download failed ({url}), trying HTML")
            return self._process_html_url(course, url)
        return self._process_html_url(course, url)

    def run(self, country, degree, courses, university=None):
        try:
            for c in courses:
                self._log(f"--- Processing '{c}' ---")
                if university:
                    res = self._search_university_domain(c, degree, university, country)
                    if res:
                        url, url_type = res
                        self._log(f"{c}: university search → {url_type} {url[:80]}")
                        f = self._process_url(c, url, url_type)
                        if f:
                            self.course_features[c] = f
                            self._log(f"{c}: SUCCESS (university)")
                            continue
                    else:
                        self._log(f"{c}: no university-domain result")

                pdf_url = self._search_course_outline(c, degree, country, university)
                if pdf_url:
                    self._log(f"{c}: country PDF → {pdf_url[:80]}")
                    f = self._process_url(c, pdf_url, 'pdf')
                    if f:
                        self.course_features[c] = f
                        self._log(f"{c}: SUCCESS (country PDF)")
                        continue

                try:
                    with DDGS() as ddgs:
                        for r in list(ddgs.text(f"{c} syllabus filetype:pdf", max_results=3)):
                            url = (r.get('href') or '').strip()
                            if not url or '.pdf' not in url.lower():
                                continue
                            self._log(f"{c}: generic PDF → {url[:80]}")
                            f = self._process_url(c, url, 'pdf')
                            if f:
                                self.course_features[c] = f
                                self._log(f"{c}: SUCCESS (generic PDF)")
                                break
                except Exception as e:
                    self._log(f"{c}: generic search error: {e}")

                if c not in self.course_features:
                    self._log(f"{c}: FALLBACK semantic")
                    f = self._apply_semantic_fallback(c)
                    f["source_url"] = "Semantic estimation (no content found)"
                    self.course_features[c] = f

            self.course_features = self._ensure_distinct_features(self.course_features)
            self.is_ready = True
        except Exception as e:
            self.error = e
            self.is_ready = True
            self._log(f"FATAL: {e}")


# SECTION 6: FSLSM AGENT
class FSLSMAgent:
    def __init__(self, llm):
        self.llm = llm
        self.answers = []

    def compute_profile(self):
        s = {"ACT_REF": 0, "SEN_INT": 0, "VIS_VER": 0, "SEQ_GLO": 0}
        for dim, d in self.answers:
            s[dim] += d
        visual = max(0.0, min(1.0, (s["VIS_VER"] + 11) / 22.0))
        sensing = max(0.0, min(1.0, (s["SEN_INT"] + 11) / 22.0))
        active = max(0.0, min(1.0, (s["ACT_REF"] + 11) / 22.0))
        glob = max(0.0, min(1.0, (s["SEQ_GLO"] + 11) / 22.0))
        return {"visual": visual, "sensing": sensing, "active": active, "global": glob,
                "verbal": 1.0 - visual, "intuitive": 1.0 - sensing,
                "reflective": 1.0 - active, "sequential": 1.0 - glob}


def get_dominant_poles(profile):
    dims = {
        "Input": ("Visual", profile["visual"], "Verbal", profile["verbal"]),
        "Perception": ("Sensing", profile["sensing"], "Intuitive", profile["intuitive"]),
        "Processing": ("Active", profile["active"], "Reflective", profile["reflective"]),
        "Understanding": ("Global", profile["global"], "Sequential", profile["sequential"])
    }
    return {d: (p1, s1) if s1 >= s2 else (p2, s2) for d, (p1, s1, p2, s2) in dims.items()}


# SECTION 7: MULTI-COURSE GRAPH
def visualize_multi_course_graph(profile, course_features, comparison_results, student_name,
                                  student_country=None, student_university=None):
    G = nx.Graph()
    G.add_node(student_name, type="student")
    dom = get_dominant_poles(profile)
    dl = {
        "Visual": f"{dom['Input'][0]} ({dom['Input'][1]:.2f})",
        "Sensing": f"{dom['Perception'][0]} ({dom['Perception'][1]:.2f})",
        "Active": f"{dom['Processing'][0]} ({dom['Processing'][1]:.2f})",
        "Global": f"{dom['Understanding'][0]} ({dom['Understanding'][1]:.2f})"
    }
    ps = {"Visual": profile["visual"], "Sensing": profile["sensing"],
          "Active": profile["active"], "Global": profile["global"]}
    
    # FIX: Map node keys to their dominant score for display
    display_scores = {
        "Visual": dom['Input'][1],
        "Sensing": dom['Perception'][1],
        "Active": dom['Processing'][1],
        "Global": dom['Understanding'][1]
    }
    
    for p, lbl in dl.items():
        G.add_node(p, type="pole", score=display_scores[p], label=lbl)
        G.add_edge(student_name, p, weight=ps[p], style="solid", color="gray")
        
    sc = {"regional": "#2ECC71", "international": "#3498DB", "semantic": "#F1C40F"}
    color_map = {}
    for cname, feats in course_features.items():
        st, _ = detect_course_source(feats.get('source_url', 'Unknown'),
                                     student_country or "Other", student_university)
        color = sc.get(st, "#FF5722")
        color_map[cname] = color
        G.add_node(cname, type="course", color=color)
        p2f = {"Visual": "C_vis", "Sensing": "C_prac", "Global": "C_struct"}
        for p, fk in p2f.items():
            if fk in feats:
                ms = profile.get(p.lower(), 0.5) * feats[fk]
                if ms > 0.01:
                    G.add_edge(p, cname, weight=ms, style="dashed", color=color)
        a_score = profile.get("active", 0.5) * 0.5
        if a_score > 0.01:
            G.add_edge("Active", cname, weight=a_score, style="dashed", color=color)
    pos = {student_name: (0, 0)}
    poles = ["Visual", "Sensing", "Active", "Global"]
    angles = np.linspace(0, 2*np.pi, len(poles), endpoint=False)
    for i, p in enumerate(poles):
        pos[p] = (np.cos(angles[i]), np.sin(angles[i]))
    nc = len(course_features)
    if nc > 0:
        ca = np.linspace(0, 2*np.pi, nc, endpoint=False)
        for i, c in enumerate(course_features.keys()):
            pos[c] = (2.2*np.cos(ca[i]), 2.2*np.sin(ca[i]))
    fig, ax = plt.subplots(figsize=(18, 14))
    nx.draw_networkx_nodes(G, pos, nodelist=[student_name], node_color="#2E7D32",
                          node_size=3500, edgecolors="black", linewidths=2, ax=ax)
    pole_n = [n for n in G.nodes if G.nodes[n].get("type") == "pole"]
    nx.draw_networkx_nodes(G, pos, nodelist=pole_n, node_color="#5D6D7E",
                          node_size=2200, edgecolors="black", linewidths=1.5, ax=ax)
    course_n = [n for n in G.nodes if G.nodes[n].get("type") == "course"]
    cc_list = [G.nodes[n].get("color", "#FF5722") for n in course_n]
    nx.draw_networkx_nodes(G, pos, nodelist=course_n, node_color=cc_list,
                          node_size=2500, edgecolors="black", linewidths=2, alpha=0.9, ax=ax)
    solid = [(u, v) for u, v, d in G.edges(data=True) if d.get("style") == "solid"]
    if solid:
        nx.draw_networkx_edges(G, pos, edgelist=solid, width=2.0, alpha=0.6,
                              edge_color="gray", ax=ax)
    dashed = [(u, v) for u, v, d in G.edges(data=True) if d.get("style") == "dashed"]
    for u, v in dashed:
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=2.0, alpha=0.9,
                              edge_color=G[u][v].get("color", "black"),
                              style="dashed", ax=ax)
    labels = {}
    for n in G.nodes:
        if n == student_name:
            labels[n] = student_name
        elif G.nodes[n].get("type") == "pole":
            labels[n] = G.nodes[n].get("label", n)
        else:
            labels[n] = n
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight="bold", ax=ax)
    el = {(u, v): f"{G[u][v].get('weight', 0):.2f}"
          for u, v in dashed if G[u][v].get('weight', 0) > 0.01}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=el, font_size=8,
                                 label_pos=0.5, rotate=False, ax=ax)
    for p, s in display_scores.items():
        x, y = pos[p]
        ax.text(x, y - 0.18, f"{s:.2f}", fontsize=8, ha='center', va='center',
                color='black', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    legend = [
        Line2D([0], [0], marker='o', color='w', label='Student',
               markerfacecolor='#2E7D32', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='FSLSM Poles',
               markerfacecolor='#5D6D7E', markersize=15),
    ]
    for c, clr in color_map.items():
        ite_val = comparison_results.get(c, {}).get("ite", 0)
        cred = course_features.get(c, {}).get("credit_hours", "N/A")
        legend.append(Line2D([0], [0], color=clr, lw=4, linestyle='--',
                             label=f"{c} (ITE: {ite_val:+.4f}, Credits: {cred})"))
    legend += [
        Line2D([0], [0], color=sc["regional"], lw=4, linestyle='--', label="Regional Course"),
        Line2D([0], [0], color=sc["international"], lw=4, linestyle='--', label="International Course"),
        Line2D([0], [0], color=sc["semantic"], lw=4, linestyle='--', label="Semantic Estimation"),
    ]
    ax.legend(handles=legend, loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
    ax.set_title(f"Multi-Course Cognitive Overlay: {student_name}\n"
                 f"(Green=Regional, Blue=International, Yellow=Semantic)",
                 fontsize=13, pad=20)
    ax.axis("off")
    plt.tight_layout()
    return fig


# SECTION 8: REASONING ENGINE
def iterative_reasoning_step(V, E, X, a, profile, course_features, concepts,
                             best_course, step_num, initial_a=None, course_names=None):
    frontier = [n for n in V if a.get(n, 0) > 0.05 and n not in ['Student', 'Context']]
    for node in frontier:
        if node in ['Visual', 'Sensing', 'Active', 'Global']:
            if not any(e[0] == node and e[1] == best_course for e in E):
                E.append((node, best_course, profile.get(node.lower(), 0.5) * 0.5))
        if node == best_course:
            for con in concepts:
                if not any(e[0] == node and e[1] == con for e in E):
                    E.append((node, con, 0.5))
    for n in V:
        if n in X:
            arr = np.array(X[n])
            X[n] = np.clip(arr * 0.9 + 0.1 * np.random.randn(len(arr)), 0, 1).tolist()
    new_a = {n: 0.01 for n in V}
    for u, v, p in E:
        if v in new_a:
            new_a[v] += p
    if initial_a is not None and course_names is not None:
        alpha = 0.7
        for n in new_a:
            if n in course_names:
                new_a[n] = alpha * initial_a.get(n, 0) + (1 - alpha) * new_a[n]
    total = sum(new_a.values())
    if total > 0:
        for n in new_a:
            new_a[n] /= total
    return V, E, X, new_a, f"  Step {step_num}: Refined attention over {len(frontier)} active nodes."


@dataclass
class CognitiveGraph:
    V: List[str]
    E: List[Dict]
    X: Dict[str, List[float]]
    a: Dict[str, float]
    reasoning_trace: List[str] = field(default_factory=list)
    best_course: str = None
    student_name: str = None
    country: str = None
    degree: str = None
    semester: str = None
    university: str = None

    def add_reasoning_step(self, step):
        self.reasoning_trace.append(step)


def generate_comparative_cognitive_graph(student_name, profile, courses, course_features,
                                         comparison_results, best_course, country=None,
                                         degree=None, semester=None, university=None,
                                         n_steps=3, llm=None):
    cg = CognitiveGraph(
        V=[student_name], E=[],
        X={student_name: [profile["visual"], profile["sensing"], profile["active"], profile["global"]]},
        a={student_name: 1.0}, best_course=best_course, student_name=student_name,
        country=country, degree=degree, semester=semester, university=university
    )
    cg.add_reasoning_step("System 1: Expanding context from student profile")
    cg.add_reasoning_step(f"  Profile: V={profile['visual']:.2f}, S={profile['sensing']:.2f}, "
                          f"A={profile['active']:.2f}, G={profile['global']:.2f}")
    ctx_nodes = []
    for v in [country, degree, f"Year: {semester}" if semester else None, university]:
        if v:
            cg.V.append(v)
            cg.X[v] = [0.5]*4
            cg.a[v] = 0.0
            ctx_nodes.append(v)
    cg.add_reasoning_step(f"  Context: {', '.join(ctx_nodes) if ctx_nodes else 'None'}")
    for c in ctx_nodes:
        cg.E.append({"from": student_name, "to": c, "relation": "has_context",
                     "prob": 0.9, "is_recommended_path": True})
        cg.a[c] = 0.3
    poles = ["Visual", "Sensing", "Active", "Global"]
    ps = {"Visual": profile["visual"], "Sensing": profile["sensing"],
          "Active": profile["active"], "Global": profile["global"]}
    for p in poles:
        cg.V.append(p)
        cg.X[p] = [ps[p], 0, 0, 0]
        cg.a[p] = 0.0
        cg.E.append({"from": student_name, "to": p, "relation": "has_preference",
                     "prob": ps[p], "is_recommended_path": True})
        cg.a[p] = ps[p] * 0.3
    for c in courses:
        cg.V.append(c)
        cg.X[c] = [comparison_results[c].get("predicted_grade", 0.5),
                   comparison_results[c].get("ite", 0),
                   course_features[c].get("C_vis", 0.5),
                   course_features[c].get("C_prac", 0.5)]
        cg.a[c] = 0.0
    for c in courses:
        matches = comparison_results[c].get("matches", {})
        is_best = (c == best_course)
        for ctx in ctx_nodes:
            cg.E.append({"from": ctx, "to": c, "relation": "offers_course",
                         "prob": 0.5, "is_recommended_path": is_best,
                         "ite": comparison_results[c].get("ite", 0)})
            cg.a[c] += 0.05
        for p in poles:
            ep = min(0.95, max(0.05, matches.get(p, 0.0)))
            cg.E.append({"from": p, "to": c, "relation": "matches",
                         "prob": ep, "is_recommended_path": is_best,
                         "ite": comparison_results[c].get("ite", 0)})
            cg.a[c] += ep * 0.1
    cg.add_reasoning_step("System 2: Comparing ITE values")
    for c, d in sorted(comparison_results.items(), key=lambda x: x[1]["ite"], reverse=True):
        cg.add_reasoning_step(f"  {c}: ITE = {d['ite']:.4f}")
    cg.add_reasoning_step(f"  Best course: {best_course} (ITE = {comparison_results[best_course]['ite']:.4f})")

    concepts = []
    if llm is not None:
        try:
            cp = ChatPromptTemplate.from_messages([
                SystemMessage(content="List exactly 4 core topics for this university course. Output ONLY a JSON list of strings."),
                HumanMessage(content=f"Course: {best_course}")
            ])
            cr = safe_llm_call(cp | llm | StrOutputParser(), {})
            cr = re.sub(r'```json\s*', '', cr)
            cr = re.sub(r'```\s*', '', cr)
            concepts = json.loads(cr)
            if not isinstance(concepts, list) or len(concepts) < 2:
                concepts = ["Core Concepts", "Applications", "Implementation", "Advanced Topics"]
        except Exception:
            concepts = ["Core Concepts", "Applications", "Implementation", "Advanced Topics"]
    else:
        concepts = ["Core Concepts", "Applications", "Implementation", "Advanced Topics"]
    if len(concepts) < 4:
        concepts += ["Core Concepts", "Applications", "Implementation", "Advanced Topics"][:4-len(concepts)]
    elif len(concepts) > 4:
        concepts = concepts[:4]

    for con in concepts:
        cg.V.append(con)
        cg.X[con] = [0.0]*4
        cg.a[con] = 0.0
    for con in concepts:
        cl = con.lower()
        vm = 0.8 if any(k in cl for k in
                        ["gate", "circuit", "graphics", "rendering", "animation", "visual"]) else 0.5
        pm = 0.8 if any(k in cl for k in
                        ["process", "memory", "file", "scheduling",
                         "analysis", "refactoring", "testing", "implementation"]) else 0.5
        gm = 0.8 if any(k in cl for k in
                        ["theory", "algorithm", "logic", "design", "architecture", "model"]) else 0.5
        ms = (vm * profile["visual"] + pm * profile["sensing"]
              + gm * profile["global"] + 0.5 * profile["active"]) / 4.0
        ep = min(0.95, max(0.05, ms))
        cg.E.append({"from": best_course, "to": con, "relation": "teaches_concept",
                     "prob": ep, "is_recommended_path": True})
        cg.a[con] = ep * 0.5
    cg.a[student_name] = 0.08
    for c in ctx_nodes:
        cg.a[c] = 0.04
    for p in poles:
        cg.a[p] = ps[p] * 0.04
    cg.a[best_course] = 0.35
    for con in concepts:
        for e in cg.E:
            if e["to"] == con:
                cg.a[con] = e["prob"] * 0.25
                break
    for c in courses:
        if c != best_course:
            cg.a[c] = 0.015
    total = sum(cg.a.values())
    if total > 0:
        for n in cg.V:
            cg.a[n] = cg.a.get(n, 0) / total

    E_t = [(e["from"], e["to"], e["prob"]) for e in cg.E]
    initial_a = cg.a.copy()
    cnames = list(course_features.keys())
    for s in range(1, n_steps + 1):
        cg.V, E_t, cg.X, cg.a, tr = iterative_reasoning_step(
            cg.V, E_t, cg.X, cg.a, profile, course_features, concepts,
            best_course, s, initial_a=initial_a, course_names=cnames)
        cg.add_reasoning_step(tr)
    cg.E = [{"from": u, "to": v, "prob": p, "relation": "reasoned"} for u, v, p in E_t]
    total = sum(cg.a.values())
    if total > 0:
        for n in cg.V:
            cg.a[n] = cg.a.get(n, 0) / total
    cg.add_reasoning_step(f"Final attention after {n_steps} iterations: " +
                          ", ".join(f"{n} ({v:.3f})" for n, v in
                                    sorted(cg.a.items(), key=lambda x: x[1], reverse=True)[:8]
                                    if v > 0.01))
    return cg


# SECTION 9: COMPREHENSIVE GRAPH VIZ
def visualize_comparative_cognitive_graph(cg, profile, comparison_results,
                                          title="Comprehensive Cognitive Graph",
                                          save_files=False, filename_base="cognitive_graph",
                                          pdf_mode=False):
    student_name = cg.student_name
    G = nx.DiGraph()
    for n in cg.V:
        G.add_node(n, attention=cg.a.get(n, 0.0))
    for e in cg.E:
        G.add_edge(e["from"], e["to"], prob=e["prob"],
                   is_recommended_path=e.get("is_recommended_path", False),
                   ite=e.get("ite", 0), relation=e.get("relation", ""))
    colors = {
        "student": "#E8F5E9", "student_border": "#2E7D32",
        "context": "#E3F2FD", "context_border": "#1565C0",
        "pole": "#F3E5F5", "pole_border": "#7B1FA2",
        "course": "#E3F2FD", "course_border": "#1565C0",
        "best_course": "#FFF8E1", "best_course_border": "#E65100",
        "concept": "#FFF3E0", "concept_border": "#E65100",
        "recommended_edge": "#D32F2F", "other_edge": "#1565C0",
    }
    fig, ax = plt.subplots(figsize=(20, 14), facecolor="white")
    ax.axis("off")
    plt.sca(ax)
    ctx_names = [c for c in [cg.country, cg.degree,
                              f"Year: {cg.semester}", cg.university] if c]
    poles = ["Visual", "Sensing", "Active", "Global"]
    ctx_n, pole_n, course_n, con_n = [], [], [], []
    for n in cg.V:
        if n == student_name:
            continue
        if n in ctx_names:
            ctx_n.append(n)
        elif n in poles:
            pole_n.append(n)
        elif any(e["to"] == n and e["from"] in poles for e in cg.E):
            course_n.append(n)
        else:
            con_n.append(n)
    spacing = 1.5 if pdf_mode else 1.3
    pos = {student_name: (0, 3.0)}
    for i, c in enumerate(ctx_n):
        pos[c] = ((i - (len(ctx_n) - 1) / 2) * spacing, 2.0)
    for i, p in enumerate(pole_n):
        pos[p] = ((i - (len(pole_n) - 1) / 2) * 0.9, 1.0)
    for i, c in enumerate(course_n):
        pos[c] = ((i - (len(course_n) - 1) / 2) * 1.4, 0.0)
    if cg.best_course and con_n:
        bx = pos.get(cg.best_course, (0, 0))[0]
        for i, con in enumerate(con_n):
            pos[con] = (bx + (i - (len(con_n) - 1) / 2) * 0.8, -1.0)
    dom = get_dominant_poles(profile)
    pdis = {
        "Visual": f"{dom['Input'][0]} ({dom['Input'][1]:.2f})",
        "Sensing": f"{dom['Perception'][0]} ({dom['Perception'][1]:.2f})",
        "Active": f"{dom['Processing'][0]} ({dom['Processing'][1]:.2f})",
        "Global": f"{dom['Understanding'][0]} ({dom['Understanding'][1]:.2f})"
    }
    nx.draw_networkx_nodes(G, pos, nodelist=[student_name], node_color=colors["student"],
                          node_size=3500, edgecolors=colors["student_border"], linewidths=2, ax=ax)
    if ctx_n:
        nx.draw_networkx_nodes(G, pos, nodelist=ctx_n, node_color=colors["context"],
                              node_size=2800, edgecolors=colors["context_border"], linewidths=2, ax=ax)
    if pole_n:
        nx.draw_networkx_nodes(G, pos, nodelist=pole_n, node_color=colors["pole"],
                              node_size=2400, edgecolors=colors["pole_border"], linewidths=2, ax=ax)
    best_n = [n for n in course_n if n == cg.best_course]
    other_n = [n for n in course_n if n != cg.best_course]
    if other_n:
        nx.draw_networkx_nodes(G, pos, nodelist=other_n, node_color=colors["course"],
                              node_size=2600, edgecolors=colors["course_border"], linewidths=2, ax=ax)
    if best_n:
        nx.draw_networkx_nodes(G, pos, nodelist=best_n, node_color=colors["best_course"],
                              node_size=3200, edgecolors=colors["best_course_border"], linewidths=3, ax=ax)
    if con_n:
        sizes = [max(1000, G.nodes[n].get("attention", 0.1) * 5000 + 500) for n in con_n]
        nx.draw_networkx_nodes(G, pos, nodelist=con_n, node_color=colors["concept"],
                              node_size=sizes, edgecolors=colors["concept_border"], linewidths=2, ax=ax)
    el = {}
    for u, v, d in G.edges(data=True):
        rec = d.get("is_recommended_path", False)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                              width=2.5 if rec else 1.5, alpha=0.8,
                              edge_color=colors["recommended_edge"] if rec else colors["other_edge"],
                              style="solid" if rec else "dashed",
                              arrows=True, arrowsize=18, arrowstyle='->', ax=ax)
        if v in course_n and "ite" in d and d["ite"] != 0 and u == cg.university:
            el[(u, v)] = f"ITE: {d['ite']:.4f}"
        elif d.get("prob", 0) > 0.05:
            el[(u, v)] = f"p={d['prob']:.2f}"
    if el:
        filtered = {k: v for k, v in el.items()
                    if k[1] == cg.best_course or k[0] == cg.best_course}
        if len(filtered) > 40:
            filtered = dict(list(filtered.items())[:40])
        if filtered:
            nx.draw_networkx_edge_labels(G, pos, filtered,
                                         font_size=10 if pdf_mode else 8,
                                         label_pos=0.5,
                                         bbox=dict(boxstyle="round,pad=0.2",
                                                   facecolor="white", alpha=0.8),
                                         ax=ax)
    labels = {}
    for n in G.nodes:
        if n == student_name:
            labels[n] = f"{student_name}\n(V:{profile['visual']:.2f}\nS:{profile['sensing']:.2f}\nG:{profile['global']:.2f})"
        elif n in ctx_n:
            labels[n] = n
        elif n in pole_n:
            labels[n] = pdis.get(n, n)
        elif n in course_n:
            iv = comparison_results.get(n, {}).get("ite", 0)
            cred = comparison_results.get(n, {}).get("credit_hours", "N/A")
            prefix = "* " if n == cg.best_course else ""
            labels[n] = f"{prefix}{n}\nITE: {iv:.4f}\nCredits: {cred}"
        else:
            labels[n] = f"{n}\na={G.nodes[n].get('attention', 0):.2f}"
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8 if pdf_mode else 9, ax=ax)
    legend = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors["student"],
                      edgecolor=colors["student_border"], label="Student"),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors["context"],
                      edgecolor=colors["context_border"], label="Context"),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors["pole"],
                      edgecolor=colors["pole_border"], label="FSLSM Pole"),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors["course"],
                      edgecolor=colors["course_border"], label="Course"),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors["best_course"],
                      edgecolor=colors["best_course_border"], label="Recommended Course"),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors["concept"],
                      edgecolor=colors["concept_border"], label="Concept"),
        plt.Line2D([0], [0], color=colors["recommended_edge"], lw=2, label="Recommended Path"),
        plt.Line2D([0], [0], color=colors["other_edge"], lw=2, linestyle='dashed', label="Other Paths")
    ]
    ax.legend(handles=legend, loc='upper left', bbox_to_anchor=(1, 1),
             fontsize=9 if pdf_mode else 10)
    ax.set_title(f"{title}\n(Solid = Recommended | Dashed = Other | Node size = Attention)",
                 fontsize=12 if pdf_mode else 14, pad=20)
    plt.tight_layout()
    if save_files:
        plt.savefig(f"{filename_base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    return fig


# SECTION 10: CAUSAL ANALYSIS
REQUIRED_NUMERIC_COLS = ['visual', 'sensing', 'active', 'global',
                          'prior_gpa', 'C_vis', 'C_prac', 'C_struct', 'final_grade']


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    original_len = len(df)
    diag = {}
    if 'Timestamp' in df.columns:
        before = len(df)
        df = df[df['Timestamp'].astype(str).str.strip() != 'Timestamp']
        diag['header_rows_dropped'] = before - len(df)
    if 'Student_ID' in df.columns:
        before = len(df)
        df = df[df['Student_ID'].astype(str).str.strip() != 'Student_ID']
        diag['student_id_header_dropped'] = before - len(df)
    before = len(df)
    df = df.drop_duplicates()
    diag['duplicates_dropped'] = before - len(df)
    for col in REQUIRED_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    before = len(df)
    df = df.dropna(subset=[c for c in REQUIRED_NUMERIC_COLS if c in df.columns])
    diag['nan_rows_dropped'] = before - len(df)
    for col in ['visual', 'sensing', 'active', 'global',
                'C_vis', 'C_prac', 'C_struct', 'final_grade']:
        if col in df.columns:
            df[col] = df[col].clip(0.0, 1.0)
    if 'prior_gpa' in df.columns:
        df['prior_gpa'] = df['prior_gpa'].clip(0.0, 4.0)
    diag['original_len'] = original_len
    diag['cleaned_len'] = len(df)
    diag['total_dropped'] = original_len - len(df)
    return df.reset_index(drop=True), diag


def get_comparative_analysis(profile, course_features, llm, student_country="Pakistan",
                              student_university=None):
    dataset_path = os.path.join(BASE_DIR, "CogNet_Advisory_Data.csv")
    if not os.path.exists(dataset_path):
        st.error(f"Dataset `CogNet_Advisory_Data.csv` not found at {dataset_path}.")
        st.stop()

    try:
        df = pd.read_csv(dataset_path, on_bad_lines='skip', low_memory=False)
        df.rename(columns={
            'Visual': 'visual', 'Sensing': 'sensing', 'Active': 'active',
            'Global': 'global', 'GPA': 'prior_gpa',
            'Course_Taken': 'course_taken', 'Final_Grade': 'final_grade'
        }, inplace=True)
    except Exception as e:
        st.error(f"Error reading the dataset: {e}")
        st.stop()
    for col in REQUIRED_NUMERIC_COLS:
        if col not in df.columns:
            st.error(f"Column '{col}' not found in dataset.")
            st.stop()
    df, diag = clean_dataset(df)

    with st.expander("🔍 Dataset Diagnostic Information", expanded=False):
        st.write(f"**Rows**: {diag['original_len']} → **{diag['cleaned_len']}** "
                 f"({diag['total_dropped']} dropped)")
        for k, label in [('header_rows_dropped', 'Corrupt embedded header rows'),
                         ('student_id_header_dropped', 'Corrupt Student_ID header rows'),
                         ('duplicates_dropped', 'Duplicate rows'),
                         ('nan_rows_dropped', 'Rows with NaN modeling values')]:
            if diag.get(k):
                st.write(f"- {label}: **{diag[k]}**")
        st.write(f"- `C_vis` range: {df['C_vis'].min():.3f} – {df['C_vis'].max():.3f} "
                 f"(std = {df['C_vis'].std():.4f})")
        st.write(f"- `final_grade` range: {df['final_grade'].min():.3f} – "
                 f"{df['final_grade'].max():.3f}")
        if df['C_vis'].std() < 0.05:
            st.warning("⚠ `C_vis` has very low variation. DML may produce attenuated estimates.")

    avg_vis = df['C_vis'].mean()
    avg_prac = df['C_prac'].mean()
    avg_struct = df['C_struct'].mean()

    X = df[['visual', 'sensing', 'active', 'global']].values
    T = df['C_vis'].values
    Y = df['final_grade'].values
    W = df[['prior_gpa']].values

    dml = None
    dml_model_name = "None"
    rscore_val = None
    try:
        dml = CausalForestDML(
            model_y=RandomForestRegressor(n_estimators=100, min_samples_leaf=10,
                                          max_depth=5, random_state=42, n_jobs=-1),
            model_t=RandomForestRegressor(n_estimators=100, min_samples_leaf=10,
                                          max_depth=5, random_state=42, n_jobs=-1),
            discrete_treatment=False, cv=3, n_estimators=300,
            min_samples_leaf=20, max_depth=5, random_state=42, inference=True)
        dml.fit(Y, T, X=X, W=W)
        dml_model_name = "CausalForestDML"
    except Exception as e:
        st.warning(f"CausalForestDML failed ({e}). Trying LinearDML.")
        try:
            dml = LinearDML(
                model_y=GradientBoostingRegressor(),
                model_t=GradientBoostingRegressor(),
                discrete_treatment=False, cv=5)
            dml.fit(Y, T, X=X, W=W)
            dml_model_name = "LinearDML"
        except Exception as e2:
            st.error(f"Both DML models failed. ({e2})")
            dml = None
            dml_model_name = "Legacy"

    if dml is not None:
        try:
            scorer = RScorer(
                model_y=RandomForestRegressor(n_estimators=50, max_depth=5,
                                              min_samples_leaf=10, random_state=42, n_jobs=-1),
                model_t=RandomForestRegressor(n_estimators=50, max_depth=5,
                                              min_samples_leaf=10, random_state=42, n_jobs=-1),
                discrete_treatment=False, cv=3)
            scorer.fit(Y, T, X=X, W=W)
            rscore_val = scorer.score(dml)
        except Exception:
            rscore_val = None

    student_X = np.array([[profile['visual'], profile['sensing'],
                           profile['active'], profile['global']]])
    if dml is not None:
        cate = float(dml.effect(student_X)[0])
        try:
            lo, hi = dml.effect_interval(student_X, alpha=0.05)
            cate_lo, cate_hi = float(lo[0]), float(hi[0])
        except Exception:
            cate_lo, cate_hi = cate, cate
    else:
        cate, cate_lo, cate_hi = 0.0, 0.0, 0.0

    if rscore_val is not None:
        if rscore_val < 0:
            st.warning(f"⚠ **R-Score = {rscore_val:.4f}** (negative). "
                       f"CATE model may be overfitting; treat estimates as directional.")
        else:
            st.info(f"✓ R-Score = {rscore_val:.4f}.")
    st.caption(f"CATE model: **{dml_model_name}** | CATE = {cate:+.4f} "
               f"(95% CI: [{cate_lo:+.4f}, {cate_hi:+.4f}])")

    results = {}
    best_course, best_ite = None, -np.inf
    for cname, feats in course_features.items():
        legacy_y = (0.5 + 0.3 * profile['visual'] * feats.get('C_vis', 0.5)
                    + 0.2 * profile['sensing'] * feats.get('C_prac', 0.5)
                    + 0.1 * profile['global'] * feats.get('C_struct', 0.5))
        legacy_baseline = (0.5 + 0.3 * profile['visual'] * avg_vis
                           + 0.2 * profile['sensing'] * avg_prac
                           + 0.1 * profile['global'] * avg_struct)
        legacy_ite = legacy_y - legacy_baseline
        delta_vis = feats.get('C_vis', 0.5) - avg_vis
        causal_ite = cate * delta_vis
        if abs(causal_ite) < 1e-6:
            causal_ite = 0.0
        predicted_grade = max(0, min(1, legacy_y + causal_ite))
        src_url = feats.get('source_url', 'Unknown')
        src_type, src_label = detect_course_source(src_url, student_country, student_university)
        results[cname] = {
            'ite': causal_ite, 'cate': cate, 'cate_ci': (cate_lo, cate_hi),
            'legacy_ite': legacy_ite, 'predicted_grade': predicted_grade,
            'credit_hours': feats.get('credit_hours', 'Unknown'),
            'source_type': src_type, 'source_label': src_label,
            'source_url': src_url, 'rscore': rscore_val,
        }
        if causal_ite > best_ite:
            best_ite, best_course = causal_ite, cname

    ite_vals = [d['ite'] for d in results.values()]
    all_eq = all(abs(i - ite_vals[0]) < 0.0005 for i in ite_vals)
    all_near_zero = all(abs(i) < 0.005 for i in ite_vals)

    def describe_course(f):
        v, p, s = f.get('C_vis', 0.5), f.get('C_prac', 0.5), f.get('C_struct', 0.5)
        cr = f.get('credit_hours', 'Unknown')
        if p > 0.7:
            nat = "practical-heavy (hands-on, lab-based)"
        elif v > 0.7:
            nat = "visual-heavy (theory with diagrams, visual aids)"
        elif s > 0.7:
            nat = "structured (well-organised, step-by-step)"
        else:
            nat = "balanced in content delivery"
        lab = "with a lab component" if str(cr).endswith("+1") else "without a lab component"
        return nat, lab, cr

    best_nat, best_lab, best_cred = describe_course(course_features[best_course])
    profile_text = (f"Visual: {profile['visual']:.2f}, Sensing (Practical): {profile['sensing']:.2f}, "
                    f"Active: {profile['active']:.2f}, Global: {profile['global']:.2f}")
    best_src_label = results[best_course].get('source_label', '')
    best_src_type = results[best_course].get('source_type', 'semantic')
    best_feats = course_features[best_course]
    comp = get_course_content_comparison(best_course, best_src_type, best_feats)
    resources = get_learning_resources(best_course)
    res_text = "\n".join(f"- {p}: {', '.join(c)}" for p, c in resources.items())

    note = ""
    if all_near_zero:
        note = (f"All causal ITE estimates are very close to zero (range "
                f"[{min(ite_vals):+.4f}, {max(ite_vals):+.4f}], 95% CI for CATE = "
                f"[{cate_lo:+.4f}, {cate_hi:+.4f}]). After controlling for prior GPA and "
                f"FSLSM dimensions, visual content of the available courses has minimal "
                f"causal impact on final grades. The recommendation is therefore driven primarily "
                f"by alignment between your FSLSM profile and each course's pedagogical features.")
    elif all_eq:
        note = (f"All courses have very similar Causal ITE values (≈ {ite_vals[0]:+.4f}). "
                f"Causal benefit is comparable; recommendation is based on best learning-style "
                f"match and content alignment.")
    elif best_ite < 0:
        note = (f"Causal ITE is negative ({best_ite:+.4f}), suggesting a modest predicted decrease "
                f"in grade relative to the average course. However, this is the least negative "
                f"option and aligns best with your learning style.")
    elif abs(best_ite) < 0.005:
        note = (f"Causal ITE is approximately zero ({best_ite:+.4f}), indicating neutral expected "
                f"impact. Recommendation is driven by optimal learning-style alignment.")
    else:
        note = f"Causal ITE is positive ({best_ite:+.4f}), suggesting a modest expected improvement."

    try:
        cp = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                f"You are a learning analytics advisor. Causal analysis ({dml_model_name}, "
                f"adjusting for prior GPA) determined best course '{best_course}' "
                f"with Causal ITE = {best_ite:+.4f}.\n\n"
                f"Model diagnostics: R-Score = "
                f"{f'{rscore_val:.4f}' if rscore_val is not None else 'N/A'}, "
                f"CATE = {cate:+.4f} (95% CI: [{cate_lo:+.4f}, {cate_hi:+.4f}]).\n\n"
                f"Student Profile: {profile_text}\n"
                f"Best course: Credits = {best_cred}, {best_nat}, {best_lab}. "
                f"Source: {best_src_label}\n\n"
                f"Content comparison: {json.dumps(comp)}\n"
                f"Resources: {res_text}\n\n"
                f"IMPORTANT NOTE:\n{note}\n\n"
                f"Write a concise but comprehensive explanation (4-6 sentences) covering: "
                f"why '{best_course}' is optimal (interpret ITE honestly), fit to learning style, "
                f"source type and content comparison, supplementary platforms, confounders "
                f"(prior GPA), and be transparent if ITE is near-zero. Use numbers."
            )),
            HumanMessage(content=f"Student: {profile_text}\n"
                                f"Results: {json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'cate_ci'} for k, v in results.items()}, indent=2)}\n"
                                f"Best: {best_course} (ITE={best_ite:+.4f}, Grade={results[best_course]['predicted_grade']:.3f})\n"
                                f"Note: {note}")
        ])
        final_exp = safe_llm_call(cp | llm | StrOutputParser(), {})
    except Exception:
        final_exp = (f"Based on causal analysis ({dml_model_name}), **{best_course}** is recommended "
                     f"(Causal ITE = {best_ite:+.4f}). {note} Course is {best_nat} and {best_lab}, "
                     f"fitting your profile ({profile_text}). Supplementary: "
                     f"{', '.join(resources.keys())}.")

    return best_course, results, final_exp, df


# SECTION 11: CROSS-VALIDATION
def run_cross_validation_with_baselines(df, n_splits=5):
    feature_cols = ['visual', 'sensing', 'active', 'global', 'prior_gpa',
                    'C_vis', 'C_prac', 'C_struct']
    X = df[feature_cols].values
    y = df['final_grade'].values
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {
        'Linear Regression': {'rmse': [], 'mae': [], 'r2': []},
        'Ridge Regression': {'rmse': [], 'mae': [], 'r2': []},
        'Random Forest': {'rmse': [], 'mae': [], 'r2': []},
        'Double ML': {'rmse': [], 'mae': [], 'r2': []},
    }
    dml_errors = []
    for fold, (tr, te) in enumerate(kf.split(X)):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        for name, model in [('Linear Regression', LinearRegression()),
                            ('Ridge Regression', Ridge(alpha=1.0, random_state=42))]:
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            results[name]['rmse'].append(np.sqrt(mean_squared_error(yte, pred)))
            results[name]['mae'].append(mean_absolute_error(yte, pred))
            results[name]['r2'].append(r2_score(yte, pred))
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(Xtr, ytr)
        prf = rf.predict(Xte)
        results['Random Forest']['rmse'].append(np.sqrt(mean_squared_error(yte, prf)))
        results['Random Forest']['mae'].append(mean_absolute_error(yte, prf))
        results['Random Forest']['r2'].append(r2_score(yte, prf))
        try:
            Xdml_tr = df.iloc[tr][['visual', 'sensing', 'active', 'global']].values
            Ttr = df.iloc[tr]['C_vis'].values
            Ytr = df.iloc[tr]['final_grade'].values
            Wtr = df.iloc[tr][['prior_gpa']].values
            if np.std(Ttr) < 1e-6:
                raise ValueError(f"Fold {fold}: C_vis constant")
            dml = LinearDML(
                model_y=GradientBoostingRegressor(n_estimators=50, random_state=42),
                model_t=GradientBoostingRegressor(n_estimators=50, random_state=42),
                discrete_treatment=False, cv=2)
            dml.fit(Ytr, Ttr, X=Xdml_tr, W=Wtr)
            Xdml_te = df.iloc[te][['visual', 'sensing', 'active', 'global']].values
            cate_te = dml.effect(Xdml_te)
            avg_vis = df['C_vis'].mean()
            base = Ridge(alpha=1.0, random_state=42).fit(Xtr, ytr)
            pred_base = base.predict(Xte)
            pred_dml = []
            for pos, (_, row) in enumerate(df.iloc[te].iterrows()):
                adj = cate_te[pos] * (row['C_vis'] - avg_vis)
                pred_dml.append(np.clip(pred_base[pos] + adj, 0, 1))
            pred_dml = np.array(pred_dml)
            results['Double ML']['rmse'].append(np.sqrt(mean_squared_error(yte, pred_dml)))
            results['Double ML']['mae'].append(mean_absolute_error(yte, pred_dml))
            results['Double ML']['r2'].append(r2_score(yte, pred_dml))
        except Exception as e:
            dml_errors.append(f"Fold {fold}: {type(e).__name__} – {str(e)}")
            continue
    summary = {}
    for m, met in results.items():
        if met['rmse']:
            summary[m] = {
                'RMSE (↓)': f"{np.mean(met['rmse']):.4f} ± {np.std(met['rmse']):.4f}",
                'MAE (↓)': f"{np.mean(met['mae']):.4f} ± {np.std(met['mae']):.4f}",
                'R2 (↑)': f"{np.mean(met['r2']):.4f} ± {np.std(met['r2']):.4f}",
            }
        else:
            summary[m] = {'RMSE (↓)': 'N/A', 'MAE (↓)': 'N/A', 'R2 (↑)': 'N/A'}
    if dml_errors:
        st.warning("Double ML failed on some folds:")
        for err in dml_errors:
            st.write(f"- {err}")
    return pd.DataFrame(summary).T


# SECTION 12: PDF REPORT
def generate_pdf_report(profile, comparison_results, explanation, cg, simple_fig,
                        student_name, country, degree, semester, university,
                        courses, course_features, logo_path=None):
    buffer = BytesIO()
    TOTAL_PAGES = 7

    def new_page():
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        return fig, ax

    def footer(fig, n):
        plt.figtext(0.5, 0.01, f"Page {n} of {TOTAL_PAGES}", ha='center', fontsize=9)

    def draw_table(ax, y, title, headers, data, col_widths,
                   title_fs=12, data_fs=8, bbox_w=0.9, row_h=0.038):
        ax.text(0.5, y, title, fontsize=title_fs, fontweight='bold',
                ha='center', va='top', transform=ax.transAxes)
        ty = y - 0.04
        n_rows = len(data) + 1
        table_h = max(0.10, row_h * n_rows)
        colours = [['#d3d3d3'] * len(headers)] + \
                  [['white'] * len(headers) for _ in range(len(data))]
        t = ax.table(cellText=[headers] + data, loc='center',
                     bbox=[0.5 - bbox_w/2, ty - table_h, bbox_w, table_h],
                     cellColours=colours, colWidths=col_widths)
        t.auto_set_font_size(False)
        t.set_fontsize(data_fs)
        t.scale(1, 1.4)
        for c in range(len(headers)):
            t[(0, c)].set_text_props(fontweight='bold', ha='center')
        return ty - table_h - 0.05

    with PdfPages(buffer) as pdf:
        # ---------------- PAGE 1: Header + Profile + Recommendation ----------------
        fig, ax = new_page()
        lp = False
        if logo_path and os.path.exists(logo_path):
            try:
                img = Image.open(logo_path).convert('RGB')
                img.thumbnail((120, 120))
                ax.imshow(img, extent=[0.05, 0.05 + 120/850, 0.92, 0.92 + 120/1100],
                          aspect='auto', zorder=10, transform=ax.transAxes)
                lp = True
            except Exception:
                pass
        ys = 0.88 if lp else 0.95
        ax.text(0.05, ys, "CogNet Advisor Report", fontsize=18, fontweight='bold',
                ha='left', va='top', transform=ax.transAxes)
        ax.text(0.05, ys - 0.04,
                "Cognitive Graph-Based Personalized Course Recommendation using FSLSM",
                fontsize=11, ha='left', va='top', transform=ax.transAxes)
        ax.text(0.05, ys - 0.08, "="*70, fontsize=10, ha='left', va='top',
                transform=ax.transAxes)
        iy = ys - 0.15
        for i, line in enumerate([f"Student: {student_name}", f"Country: {country}",
                                  f"Degree: {degree}", f"Year: {semester}",
                                  f"University: {university if university else 'N/A'}"]):
            ax.text(0.05, iy - i*0.04, line, fontsize=11, ha='left', va='top',
                    transform=ax.transAxes)
        yp = iy - 5*0.04 - 0.06

        prof_data = [[k.title(), f"{profile[k]:.3f}"]
                     for k in ['visual', 'verbal', 'sensing', 'intuitive',
                               'active', 'reflective', 'global', 'sequential']]
        yp = draw_table(ax, yp, "LEARNING STYLE PROFILE",
                        ["Style", "Score"], prof_data, [0.5, 0.5])

        d = comparison_results[cg.best_course]
        iv = d.get('ite', 0.0)
        if abs(iv) < 1e-6: iv = 0.0
        rec = [["Course", cg.best_course],
               ["Causal ITE", f"{iv:+.4f}"],
               ["Predicted Grade", f"{d.get('predicted_grade', 0):.3f}"],
               ["Credit Hours", d.get('credit_hours', 'Unknown')],
               ["Source", d.get('source_label', '')]]
        if d.get('rscore') is not None:
            rec.append(["R-Score (CATE model)", f"{d['rscore']:+.4f}"])
        draw_table(ax, yp, "RECOMMENDED COURSE", ["Attribute", "Value"],
                   rec, [0.5, 0.5])
        footer(fig, 1)
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # ---------------- PAGE 2: Match + ITE ----------------
        fig, ax = new_page()
        ax.text(0.05, 0.95, "LEARNING STYLE MATCH & ITE COMPARISON",
                fontsize=14, fontweight='bold', ha='left', va='top', transform=ax.transAxes)
        bf = course_features[cg.best_course]
        match_data = [
            ["Visual", f"{profile['visual']:.2f}",
             f"{bf.get('C_vis', 0.5):.2f}",
             f"{profile['visual']*bf.get('C_vis', 0.5):.2f}"],
            ["Sensing (Practical)", f"{profile['sensing']:.2f}",
             f"{bf.get('C_prac', 0.5):.2f}",
             f"{profile['sensing']*bf.get('C_prac', 0.5):.2f}"],
            ["Global (Structured)", f"{profile['global']:.2f}",
             f"{bf.get('C_struct', 0.5):.2f}",
             f"{profile['global']*bf.get('C_struct', 0.5):.2f}"],
            ["Active (proxy)", f"{profile['active']:.2f}", "0.50",
             f"{profile['active']*0.5:.2f}"],
        ]
        yp = 0.88
        yp = draw_table(ax, yp, "LEARNING STYLE MATCH (Primary Poles)",
                        ["Pole", "Your Score", "Course Feature", "Match"],
                        match_data, [0.28, 0.22, 0.25, 0.25])
        ite_data = []
        for c, dd in comparison_results.items():
            m = "*" if c == cg.best_course else " "
            v = dd.get('ite', 0.0)
            if abs(v) < 1e-6: v = 0.0
            ite_data.append([f"{m} {c}", f"{v:+.4f}",
                             f"{dd.get('legacy_ite', 0):+.3f}",
                             f"{dd.get('predicted_grade', 0):.3f}"])
        draw_table(ax, yp, "COMPARATIVE ITE RESULTS",
                   ["Course", "Causal ITE", "Legacy ITE", "Grade"],
                   ite_data, [0.35, 0.20, 0.22, 0.23])
        footer(fig, 2)
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # ---------------- PAGE 3: Sources + Resources ----------------
        fig, ax = new_page()
        ax.text(0.05, 0.95, "SOURCE TRACEABILITY & LEARNING RESOURCES",
                fontsize=14, fontweight='bold', ha='left', va='top', transform=ax.transAxes)
        sd = []
        for c, dd in comparison_results.items():
            raw = dd.get('source_url', '')
            url = clean_source_url(raw)
            if url and url.lower().startswith('http'):
                url_disp = textwrap.fill(url, width=40)
                pdf_status = "Cached PDF"
            else:
                url_disp = "(semantic fallback — no URL)"
                pdf_status = "Not available"
            sd.append([c, url_disp, dd.get('source_label', ''), pdf_status])
        yp = 0.88
        yp = draw_table(ax, yp, "SCRAPED SOURCES",
                        ["Course", "Source URL", "Source Type", "PDF Status"],
                        sd, [0.22, 0.38, 0.22, 0.18])
        ax.text(0.5, yp, "SUGGESTED LEARNING RESOURCES",
                fontsize=12, fontweight='bold', ha='center', va='top', transform=ax.transAxes)
        yp -= 0.05
        resources_urls = get_learning_resources_with_urls(cg.best_course)
        for plat, courses_list in resources_urls.items():
            ax.text(0.06, yp, f"▸ {plat}", fontsize=9, fontweight='bold',
                    ha='left', va='top', transform=ax.transAxes, color='#1565C0')
            yp -= 0.028
            for cn, _ in courses_list:
                ax.text(0.10, yp, f"• {cn}", fontsize=8, ha='left', va='top',
                        transform=ax.transAxes)
                yp -= 0.024
            yp -= 0.010
        footer(fig, 3)
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # ---------------- PAGE 4: Multi-Course Overlay ----------------
        simple_fig.tight_layout()
        plt.figtext(0.5, 0.01, f"Page 4 of {TOTAL_PAGES}", ha='center', fontsize=9)
        pdf.savefig(simple_fig, dpi=150, bbox_inches='tight')
        plt.close(simple_fig)

        # ---------------- PAGE 5: Comprehensive Graph ----------------
        try:
            cf = visualize_comparative_cognitive_graph(
                cg, profile, comparison_results,
                title=f"Comprehensive Cognitive Graph: {student_name}",
                save_files=False, pdf_mode=True)
            cf.tight_layout()
            plt.figtext(0.5, 0.01, f"Page 5 of {TOTAL_PAGES}", ha='center', fontsize=9)
            pdf.savefig(cf, dpi=150, bbox_inches='tight')
            plt.close(cf)
        except Exception as e:
            fig, ax = new_page()
            ax.text(0.5, 0.5, f"Graph rendering failed: {e}\n"
                              f"See reasoning trace on Page 6.",
                    ha='center', va='center', fontsize=11, transform=ax.transAxes)
            footer(fig, 5)
            pdf.savefig(fig, dpi=150, bbox_inches='tight')
            plt.close(fig)

        # ---------------- PAGE 6: Reasoning Trace ----------------
        fig, ax = new_page()
        ax.text(0.05, 0.95, "REASONING TRACE",
                fontsize=14, fontweight='bold', ha='left', va='top', transform=ax.transAxes)
        trace_lines = []
        for line in cg.reasoning_trace:
            for wrapped in textwrap.wrap(line, width=105):
                trace_lines.append(wrapped)
        trace_text = "\n".join(trace_lines[:45])
        ax.text(0.05, 0.90, trace_text, fontsize=7.5, ha='left', va='top',
                transform=ax.transAxes, fontfamily='monospace')
        footer(fig, 6)
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # ---------------- PAGE 7: Master Explanation ----------------
        fig, ax = new_page()
        ax.text(0.05, 0.95, "MASTER CAUSAL EXPLANATION",
                fontsize=14, fontweight='bold', ha='left', va='top', transform=ax.transAxes)
        wrapped_expl = "\n".join(textwrap.wrap(explanation, width=95))
        ax.text(0.05, 0.88, wrapped_expl, fontsize=10, ha='left', va='top',
                transform=ax.transAxes)
        footer(fig, 7)
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close(fig)

    buffer.seek(0)
    return buffer


# SECTION 13: MAIN APP
def main():
    st.set_page_config(page_title="CogNet Advisor", layout="wide")

    if not check_password():
        st.stop()

    logo_exists = LOGO_PATH and os.path.exists(LOGO_PATH)
    c1, c2 = st.columns([1, 5])
    with c1:
        if logo_exists:
            st.image(LOGO_PATH, width=120)
    with c2:
        st.title("CogNet Advisor")
        st.markdown("*Cognitive Graph-Based Personalized Course Recommendation System using FSLSM*")

    with st.sidebar:
        st.header("API Key")
        saved = load_api_key()
        if saved:
            st.success("API key loaded.")
            if st.button("Clear saved API key"):
                clear_api_key()
                st.rerun()
            api_key = saved
        else:
            api_key = st.text_input("OpenAI API Key", type="password")
            remember = st.checkbox("Remember my API key")
            if api_key and remember:
                save_api_key(api_key)
                st.success("Key saved.")
        if not api_key:
            st.warning("Please enter your OpenAI API key.")
            st.stop()
        os.environ["OPENAI_API_KEY"] = api_key

        st.header("Student Profile")
        name = st.text_input("Name", "Student")
        country = st.selectbox("Country", ["Pakistan", "USA", "UK", "Canada", "Australia",
                                          "Germany", "France", "India", "China", "Japan",
                                          "South Korea", "Singapore", "Malaysia", "UAE",
                                          "Saudi Arabia", "Egypt", "South Africa", "Brazil",
                                          "Mexico", "Italy", "Spain", "Netherlands",
                                          "Sweden", "Norway", "Russia", "Other"])
        degree = st.selectbox("Degree", ["BS Computer Science", "BS Artificial Intelligence",
                                        "BS Software Engineering", "BS Data Science",
                                        "BS Cybersecurity", "MS Computer Science",
                                        "MS Data Science", "MS Artificial Intelligence",
                                        "PhD Computer Science", "BBA", "MBA", "Other"])
        semester = st.selectbox("Academic Year", list(range(2020, 2031)), index=6)
        university = st.text_input("University (optional)")

        st.header("Courses")
        course1 = st.text_input("Course 1", "Artificial Intelligence")
        course2 = st.text_input("Course 2", "Computer Networks")
        course3 = st.text_input("Course 3", "Data Science")
        courses = [c for c in [course1, course2, course3] if c.strip()]
        if not courses:
            st.warning("Please enter at least one course.")
            st.stop()
        if st.button("Start Assessment", type="primary"):
            st.session_state["start_assessment"] = True

    llm, active_model = build_llm()
    st.sidebar.caption(f"🧠 Active model: `{active_model}`")
    embeddings = OpenAIEmbeddings()

    for k, v in [("start_assessment", False), ("answers", []),
                 ("agent", FSLSMAgent(llm)), ("profile", None),
                 ("course_features", None), ("comparison_results", None),
                 ("best_course", None), ("explanation", None), ("cg", None),
                 ("analysis_started", False), ("simple_fig", None),
                 ("pdf_buffer", None), ("rag_log", [])]:
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state.start_assessment and st.session_state.profile is None:
        st.subheader("FSLSM Questionnaire")
        st.markdown("Please answer all 44 questions below.")
        answers = {}
        with st.form(key="fslsm_form"):
            for idx, (qt, dim, pole) in enumerate(FSLSM_QUESTIONS):
                oa, ob = extract_options(qt)
                clean = re.sub(r'\s*\(a\)\s*.*?\s*\(b\)\s*.*', '', qt).strip()
                choice = st.radio(f"Q{idx+1}: {clean}",
                                  [f"(a) {oa}", f"(b) {ob}"],
                                  key=f"q_{idx}", index=None)
                answers[idx] = choice
            sub = st.form_submit_button("Submit All Answers")
        if sub:
            if any(answers.get(i) is None for i in range(len(FSLSM_QUESTIONS))):
                st.warning("Please answer all questions.")
            else:
                for idx, (_, dim, pole) in enumerate(FSLSM_QUESTIONS):
                    sel = "a" if answers[idx].startswith("(a)") else "b"
                    d = 1 if sel == pole else -1
                    st.session_state.answers.append((dim, d))
                st.session_state.agent.answers = st.session_state.answers
                st.session_state.profile = st.session_state.agent.compute_profile()
                st.success("Assessment complete!")
                if st.button("Analyze"):
                    st.session_state.analysis_started = True
                    st.rerun()
    else:
        if st.session_state.profile is not None and st.session_state.best_course is not None:
            st.header("Results")
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                st.subheader("Learning Style Profile")
                for k in ['visual', 'sensing', 'active', 'global']:
                    st.write(f"**{k.title()}**: {st.session_state.profile[k]:.3f}")
            with cc2:
                st.subheader(" ")
                for k in ['verbal', 'intuitive', 'reflective', 'sequential']:
                    st.write(f"**{k.title()}**: {st.session_state.profile[k]:.3f}")
            with cc3:
                st.subheader("Recommended Course")
                st.success(f"**{st.session_state.best_course}**")
                d = st.session_state.comparison_results[st.session_state.best_course]
                st.write(f"Causal ITE: {d['ite']:+.4f}")
                st.write(f"Predicted Grade: {d['predicted_grade']:.3f}")
                if d.get('rscore') is not None:
                    st.write(f"R-Score: {d['rscore']:+.4f}")

            st.subheader("Learning Style Match (Primary Poles)")
            bf = st.session_state.course_features[st.session_state.best_course]
            match_df = pd.DataFrame([
                {"Pole": "Visual", "Your Score": f"{st.session_state.profile['visual']:.2f}",
                 "Course Feature (C_vis)": f"{bf.get('C_vis', 0.5):.2f}",
                 "Match": f"{st.session_state.profile['visual']*bf.get('C_vis', 0.5):.2f}"},
                {"Pole": "Sensing (Practical)", "Your Score": f"{st.session_state.profile['sensing']:.2f}",
                 "Course Feature (C_prac)": f"{bf.get('C_prac', 0.5):.2f}",
                 "Match": f"{st.session_state.profile['sensing']*bf.get('C_prac', 0.5):.2f}"},
                {"Pole": "Global (Structured)", "Your Score": f"{st.session_state.profile['global']:.2f}",
                 "Course Feature (C_struct)": f"{bf.get('C_struct', 0.5):.2f}",
                 "Match": f"{st.session_state.profile['global']*bf.get('C_struct', 0.5):.2f}"},
                {"Pole": "Active (proxy)", "Your Score": f"{st.session_state.profile['active']:.2f}",
                 "Course Feature (proxy)": "0.50",
                 "Match": f"{st.session_state.profile['active']*0.5:.2f}"},
            ])
            st.dataframe(match_df, use_container_width=True, hide_index=True)

            st.subheader("Comparative ITE Results")
            for c, dd in st.session_state.comparison_results.items():
                raw = dd.get('source_url', '')
                src = clean_source_url(raw)
                if src.lower().startswith('http'):
                    src_disp = src[:80]
                    pdf_status = "Cached PDF"
                else:
                    src_disp = "(semantic fallback — no URL)"
                    pdf_status = "Not available"
                iv = dd['ite']
                if abs(iv) < 1e-6: iv = 0.0
                st.write(f"**{c}**: Causal ITE = {iv:+.4f} "
                         f"(Leg
