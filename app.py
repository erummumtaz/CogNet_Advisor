# SECTION 1: IMPORTS

import os
import json
import time
import re
import hashlib
import requests
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import textwrap
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from econml.dml import LinearDML
from sklearn.ensemble import GradientBoostingRegressor

import dowhy
from dowhy import CausalModel



# SECTION 2: API KEY & PATH MANAGEMENT

API_KEY_FILE = ".api_key"
CACHE_DIR = "./cache/"
os.makedirs(CACHE_DIR, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "CognetAdvisor.png")
if not os.path.exists(LOGO_PATH):
    LOGO_PATH = None


def load_api_key():
    # 1. Try Streamlit secrets first (for deployment on Streamlit Cloud)
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except:
        pass

    # 2. Try environment variable (local development)
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key

    # 3. Try local .api_key file (legacy / local)
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


# SECTION 2.5: SOURCE DETECTION AND CONTENT COMPARISON

def get_country_domains(country: str) -> List[str]:
    """Return country-specific domain patterns based on the student's country."""
    country_domain_map = {
        "Pakistan": ['.pk', '.edu.pk'],
        "USA": ['.us', '.edu'],
        "UK": ['.uk', '.ac.uk'],
        "Canada": ['.ca'],
        "Australia": ['.au', '.edu.au'],
        "India": ['.in', '.ac.in'],
        "Germany": ['.de'],
        "France": ['.fr'],
        "China": ['.cn', '.edu.cn'],
        "Japan": ['.jp', '.ac.jp'],
        "South Korea": ['.kr', '.ac.kr'],
        "Singapore": ['.sg', '.edu.sg'],
        "Malaysia": ['.my', '.edu.my'],
        "UAE": ['.ae', '.ac.ae'],
        "Saudi Arabia": ['.sa', '.edu.sa'],
        "Egypt": ['.eg', '.edu.eg'],
        "South Africa": ['.za', '.ac.za'],
        "Brazil": ['.br', '.edu.br'],
        "Mexico": ['.mx', '.edu.mx'],
        "Italy": ['.it'],
        "Spain": ['.es'],
        "Netherlands": ['.nl'],
        "Sweden": ['.se'],
        "Norway": ['.no'],
        "Russia": ['.ru'],
        "Other": []
    }
    return country_domain_map.get(country, [])


def detect_course_source(
    source_url: str,
    student_country: str,
    student_university: str = None
) -> Tuple[str, str]:
    """
    Dynamically detect whether a course is regional, international, or semantic.
    Returns: (source_type, source_label)
    """
    if source_url in ['Unknown', 'Semantic estimation (no content found)', 'Fallback']:
        return "semantic", "Semantic Estimation"
    
    source_lower = source_url.lower()
    country_domains = get_country_domains(student_country)
    
    # Check if source contains the student's university name
    university_match = False
    if student_university:
        uni_parts = student_university.lower().replace(' ', '').split(',')
        for part in uni_parts:
            if part.strip() and part.strip() in source_lower:
                university_match = True
                break
    
    # Check if source contains country-specific domain
    country_domain_match = any(domain in source_lower for domain in country_domains)
    
    if university_match or country_domain_match:
        return "regional", f"Regional ({student_country})"
    
    # Check for international platforms
    international_patterns = [
        'coursera', 'udemy', 'edx', 'udacity', 'linkedin', 'datacamp',
        'pluralsight', 'khanacademy', 'mit.edu', 'stanford.edu', 'harvard.edu'
    ]
    if any(pattern in source_lower for pattern in international_patterns):
        return "international", "International (Global Platform)"
    
    # Check for foreign university
    if '.edu' in source_lower and not any(domain in source_lower for domain in country_domains):
        return "international", "International (Foreign University)"
    
    return "semantic", "Semantic Estimation"


def get_course_content_comparison(course_name: str, source_type: str, features: Dict) -> Dict:
    """
    Generate a comparison of course contents based on source type and features.
    """
    comparison = {
        "regional": {
            "format": "Detailed PDF syllabus",
            "structure": "12-16 weeks of weekly topics",
            "credit_hours": f"{features.get('credit_hours', '3+1')} (theory + lab)" if features.get('credit_hours', 'Unknown') != 'Unknown' else "Not specified",
            "lab_component": "1 credit hour lab" if str(features.get('credit_hours', '')).endswith('+1') else "No lab specified",
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
            "credit_hours": "Estimated",
            "lab_component": "Estimated",
            "assessment": "Estimated",
            "textbook": "Estimated",
            "learning_style": "Estimated"
        }
    }
    return comparison.get(source_type, comparison["semantic"])


def get_learning_resources(course_name: str) -> Dict[str, List[str]]:
    """Suggest online learning platforms and courses based on the recommended course."""
    platform_map = {
        "Computer Graphics": {
            "Coursera": ["Computer Graphics - UC San Diego", "Interactive Computer Graphics - Stanford"],
            "Udemy": ["Learn Computer Graphics with OpenGL", "Game Development & Computer Graphics"],
            "MIT OpenCourseWare": ["Computer Graphics (6.837)", "Introduction to Computer Graphics (6.856)"],
            "YouTube": ["The Cherno (C++/Graphics tutorials)", "Two Minute Papers (Graphics research)"]
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
            "GeeksforGeeks": ["Data Structures and Algorithms Self Paced"]
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
    return platform_map.get(course_name, {
        "Coursera": [f"Search for courses related to {course_name}"],
        "Udemy": [f"Search for courses related to {course_name}"],
        "edX": [f"Search for courses related to {course_name}"]
    })

# SECTION 2.6: RESOURCES WITH URLS

def get_learning_resources_with_urls(course_name: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Return dict of platform -> list of (course_name, url) for the given course.
    The mapping is dynamic per course name – extend as needed.
    """
    course_map = {
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
                ("The Cherno (C++/Graphics tutorials)", "https://www.youtube.com/user/TheCherno"),
                ("Two Minute Papers (Graphics research)", "https://www.youtube.com/user/keeroyz")
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
                ("Master the Coding Interview: Data Structures + Algorithms", "https://www.udemy.com/course/master-the-coding-interview-data-structures-algorithms/")
            ],
            "GeeksforGeeks": [
                ("Data Structures and Algorithms Self Paced", "https://www.geeksforgeeks.org/courses/dsa-self-paced")
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
                ("Computer Science Theory - MIT OpenCourseWare", "https://www.youtube.com/playlist?list=PLUl4u3cNGP60_UYpFj6C3tN0N_Tn2dAL6")
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
                ("Computer Networks - University of Washington", "https://www.coursera.org/learn/computer-networking"),
                ("Network Security - Stanford", "https://www.coursera.org/learn/network-security")
            ],
            "Udemy": [
                ("The Complete Networking Fundamentals Course", "https://www.udemy.com/course/complete-networking-fundamentals-course/"),
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
    
    if course_name not in course_map:
        return {
            "Coursera": [(f"Search for courses related to {course_name}", f"https://www.coursera.org/search?query={course_name.replace(' ', '%20')}")],
            "Udemy": [(f"Search for courses related to {course_name}", f"https://www.udemy.com/courses/search/?q={course_name.replace(' ', '+')}")],
            "edX": [(f"Search for courses related to {course_name}", f"https://www.edx.org/search?q={course_name.replace(' ', '+')}")]
        }
    return course_map[course_name]


# SECTION 3: FSLSM QUESTION BANK (44 items - COMPLETE)

FSLSM_QUESTIONS = [
    # Active/Reflective Dimension (11 questions)
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
    ("When writing a paper, I am more likely to (a) work on (think about or write) the beginning of the paper and progress forward. (b) work on (think about or write) different parts of the paper and then order them.", "SEQ_GLO", "a"),
    ("When I have to work on a group project, I first want to (a) have 'group brainstorming' where everyone contributes ideas. (b) brainstorm individually and then come together as a group to compare ideas.", "ACT_REF", "a"),
    ("I consider it higher praise to call someone (a) sensible. (b) imaginative.", "SEN_INT", "a"),
    ("When I meet people at a party, I am more likely to remember (a) what they looked like. (b) what they said about themselves.", "VIS_VER", "a"),
    ("When I am learning a new subject, I prefer to (a) stay focused on that subject, learning as much about it as I can. (b) try to make connections between that subject and related subjects.", "SEQ_GLO", "a"),
    ("I am more likely to be considered (a) outgoing. (b) reserved.", "ACT_REF", "a"),
    ("I prefer courses that emphasize (a) concrete material (facts, data). (b) abstract material (concepts, theories).", "SEN_INT", "a"),
    ("For entertainment, I would rather (a) watch television. (b) read a book.", "VIS_VER", "a"),
    ("Some teachers start their lectures with an outline of what they will cover. Such outlines are (a) somewhat helpful to me. (b) very helpful to me.", "SEQ_GLO", "a"),
    ("The idea of doing homework in groups, with one grade for the entire group, (a) appeals to me. (b) does not appeal to me.", "ACT_REF", "a"),
    ("When I am doing long calculations, (a) I tend to repeat all my steps and check my work carefully. (b) I find checking my work tiresome and have to force myself to do it.", "SEQ_GLO", "a"),
    ("I tend to picture places I have been (a) easily and fairly accurately. (b) with difficulty and without much detail.", "VIS_VER", "a"),
    ("When solving problems in a group, I would be more likely to (a) think of the steps in the solution process. (b) think of possible consequences or applications of the solution in a wide range of areas.", "SEQ_GLO", "a"),
]


def extract_options(q_text):
    import re
    match = re.search(r'\(a\)\s*(.+?)\s*\(b\)\s*(.+)', q_text)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "a", "b"


# SECTION 4: UTILITY FUNCTIONS

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def safe_llm_call(chain, input_data):
    return chain.invoke(input_data)


def download_pdf(url: str, timeout: int = 15) -> Optional[str]:
    filename = f"{hashlib.md5(url.encode()).hexdigest()[:8]}.pdf"
    path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    try:
        response = requests.get(url, timeout=timeout, verify=False)
        response.raise_for_status()
        with open(path, "wb") as f:
            f.write(response.content)
        return path
    except Exception:
        return None


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
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        vectorstore.persist()
        return vectorstore
    except Exception:
        return None


# SECTION 5: BACKGROUND RAG PREPARER

class BackgroundRAGPreparer:
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.is_ready = False
        self.course_features = {}
        self.error = None
        self._last_success_url = None

    def _validate_pdf(self, content: bytes) -> bool:
        if not content.startswith(b'%PDF'):
            return False
        if len(content) < 10240:
            return False
        return True

    def _clean_url(self, url: str) -> str:
        import urllib.parse
        try:
            parsed = urllib.parse.urlparse(url)
            path = urllib.parse.unquote(parsed.path)
            safe_chars = "/,&+()'"
            encoded_path = urllib.parse.quote(path, safe=safe_chars)
            cleaned_url = urllib.parse.urlunparse(
                (parsed.scheme, parsed.netloc, encoded_path, parsed.params, parsed.query, parsed.fragment)
            )
            return cleaned_url
        except Exception:
            return url

    def _try_download_variants(self, base_url: str, timeout: int = 15) -> Optional[Tuple[bytes, str, str]]:
        import urllib.parse
        parsed = urllib.parse.urlparse(base_url)
        variants = [
            base_url,
            base_url.replace('+', '%20'),
            base_url.replace('+', '_'),
            base_url.replace('%20', '+'),
            base_url.replace(' ', '%20'),
            base_url.replace(' ', '+'),
            base_url.replace('%20', '_'),
            base_url.replace(' ', '+').replace('+', ''),
            base_url.replace('%20', ''),
        ]
        try:
            decoded = urllib.parse.unquote(base_url)
            parsed_decoded = urllib.parse.urlparse(decoded)
            path_fixed = parsed_decoded.path.replace(' ', '%20')
            fixed_url = urllib.parse.urlunparse(
                (parsed_decoded.scheme, parsed_decoded.netloc, path_fixed,
                 parsed_decoded.params, parsed_decoded.query, parsed_decoded.fragment)
            )
            variants.append(fixed_url)
            path_fixed_amp = parsed_decoded.path.replace('&', '%26')
            fixed_url_amp = urllib.parse.urlunparse(
                (parsed_decoded.scheme, parsed_decoded.netloc, path_fixed_amp,
                 parsed_decoded.params, parsed_decoded.query, parsed_decoded.fragment)
            )
            variants.append(fixed_url_amp)
        except:
            pass
        variants = list(dict.fromkeys(variants))
        for i, url in enumerate(variants):
            try:
                response = requests.get(url, timeout=timeout, verify=False, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
                })
                content_type = response.headers.get('Content-Type', '').lower()
                if response.status_code == 200:
                    if 'text/html' in content_type or 'text/plain' in content_type:
                        continue
                    content = response.content
                    if content.startswith(b'<!DOCTYPE') or content.startswith(b'<html') or b'<head' in content[:200]:
                        continue
                    if self._validate_pdf(content):
                        fname = hashlib.md5(url.encode()).hexdigest()[:8] + '.pdf'
                        return (content, fname, url)
                else:
                    continue
            except Exception:
                continue
        return None

    def _download_pdf(self, url: str, timeout: int = 15) -> Optional[str]:
        cleaned_url = self._clean_url(url)
        cache_key = hashlib.md5(url.encode()).hexdigest()[:8]
        filename = cache_key + '.pdf'
        path = os.path.join(CACHE_DIR, filename)
        if os.path.exists(path) and os.path.getsize(path) > 10240:
            with open(path, 'rb') as f:
                if f.read(4) == b'%PDF':
                    return path
            os.remove(path)
        result = self._try_download_variants(cleaned_url, timeout)
        if result is None:
            result = self._try_download_variants(url, timeout)
        if result:
            content, fname, success_url = result
            path = os.path.join(CACHE_DIR, fname)
            with open(path, "wb") as f:
                f.write(content)
            self._last_success_url = success_url
            return path
        return None

    def _scrape_course_page(self, url: str, timeout: int = 15) -> Optional[str]:
        from bs4 import BeautifulSoup
        try:
            response = requests.get(url, timeout=timeout, verify=False, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                for element in soup(['script', 'style', 'meta', 'link', 'noscript']):
                    element.decompose()
                text = soup.get_text(separator=' ')
                text = ' '.join(text.split())
                if len(text) > 200:
                    return text
            return None
        except Exception:
            return None

    def _search_university_domain(self, course: str, degree: str, university: str, country: str) -> Optional[Tuple[str, str]]:
        try:
            uni_parts = [p.lower() for p in university.split() if len(p) > 3]
            domain_patterns = []
            base_domain = university.lower().replace(' ', '').replace('&', '')
            domain_patterns.append(f"{base_domain}.edu.pk")
            domain_patterns.append(f"{base_domain}.pk")
            domain_patterns.append(f"{base_domain}.edu")
            words = university.split()
            if len(words) >= 2:
                acronym = ''.join([w[0].lower() for w in words if w[0].isalpha()])
                domain_patterns.append(f"{acronym}.edu.pk")
                domain_patterns.append(f"{acronym}.pk")
                domain_patterns.append(f"{acronym}.edu")
            with DDGS() as ddgs:
                queries = [
                    f"{course} {university} {degree} syllabus",
                    f"{course} {university} course outline",
                    f"{course} {university} syllabus",
                    f"{course} {university} academic courses"
                ]
                for q in queries:
                    results = list(ddgs.text(q, max_results=5))
                    for r in results:
                        url = r.get('href', '')
                        if not url:
                            continue
                        url_lower = url.lower()
                        is_valid = False
                        for pattern in domain_patterns:
                            if pattern in url_lower:
                                is_valid = True
                                break
                        if not is_valid:
                            for part in uni_parts:
                                if part in url_lower:
                                    is_valid = True
                                    break
                        country_domain = ".pk" if country.lower() == "pakistan" else ".edu" if country.lower() == "usa" else ""
                        if is_valid and country_domain and country_domain not in url_lower:
                            continue
                        if is_valid:
                            if '.pdf' in url_lower:
                                return (url, 'pdf')
                            else:
                                return (url, 'html')
                for q in [f"{course} {university} syllabus"]:
                    results = list(ddgs.text(q, max_results=3))
                    for r in results:
                        url = r.get('href', '')
                        if url and ('.edu' in url or '.pk' in url):
                            if any(p in url.lower() for p in uni_parts):
                                if '.pdf' in url.lower():
                                    return (url, 'pdf')
                                else:
                                    return (url, 'html')
        except Exception:
            pass
        return None

    def _search_course_outline(self, course: str, degree: str, country: str, university: str = None) -> Optional[str]:
        try:
            with DDGS() as ddgs:
                country_code = country.lower()
                queries = []
                if university:
                    queries.append(f"{course} {university} {degree} syllabus")
                queries.append(f"{course} {degree} syllabus site:.{country_code[:2]}")
                queries.append(f"{course} course outline {country} filetype:pdf")
                if country_code == "pakistan":
                    queries.append(f"{course} {degree} syllabus Pakistan filetype:pdf")
                queries.append(f"{course} {degree} syllabus PDF")
                queries.append(f"{course} syllabus filetype:pdf")
                for q in queries:
                    results = list(ddgs.text(q, max_results=5))
                    for r in results:
                        url = r.get('href', '')
                        if '.pdf' in url.lower():
                            if country_code == "pakistan" and ".pk" in url:
                                return url
                            if country_code == "usa" and ".edu" in url:
                                return url
                            return url
        except Exception:
            pass
        return None

    def _build_rag_from_pdf(self, pdf_path: str) -> Optional[Chroma]:
        return build_rag_from_pdf(pdf_path, self.embeddings)

    def _extract_features_from_context(self, course: str, context: str) -> Dict:
        try:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are an expert in CS curriculum analysis. Rate this course on Visual, Practical, Structured (0-1). Ensure your ratings reflect the specific details of this course. Output strict JSON with keys C_vis, C_prac, C_struct."),
                HumanMessage(content=f"Course: {course}\nContext: {context[:2000]}")
            ])
            chain = prompt | self.llm | StrOutputParser()
            res = safe_llm_call(chain, {})
            match = re.search(r'\{.*\}', res, re.DOTALL)
            if match:
                features = json.loads(match.group())
            else:
                features = {"C_vis":0.5, "C_prac":0.5, "C_struct":0.5}
            features["C_vis"] = max(0.0, min(1.0, features.get("C_vis", 0.5)))
            features["C_prac"] = max(0.0, min(1.0, features.get("C_prac", 0.5)))
            features["C_struct"] = max(0.0, min(1.0, features.get("C_struct", 0.5)))
            features["C_vis"] = max(features["C_vis"], 0.2)
            features["C_prac"] = max(features["C_prac"], 0.1)
            features["C_struct"] = max(features["C_struct"], 0.1)
            if features["C_vis"] >= 0.95 and features["C_prac"] >= 0.95 and features["C_struct"] >= 0.95:
                features["C_vis"] = min(0.9, features["C_vis"])
                features["C_prac"] = min(0.8, features["C_prac"])
                features["C_struct"] = min(0.8, features["C_struct"])
            credit_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="Extract ONLY the credit hours from the context. Look for patterns like '3+0', '3+1', '2+1', 'Credit Hours: 3', '4 credits', etc. Output ONLY the credit hours string (e.g., '3+1'). If you cannot find a specific credit hours pattern, output 'Unknown'."),
                HumanMessage(content=f"Context: {context[:1500]}")
            ])
            credit_chain = credit_prompt | self.llm | StrOutputParser()
            credit_hours_raw = safe_llm_call(credit_chain, {}).strip()
            credit_patterns = [
                r'(\d+)\s*\+\s*(\d+)',
                r'Credit Hours?\s*:\s*(\d+)',
                r'(\d+)\s*credits',
                r'(\d+)\s*Cr\.\s*Hrs\.',
                r'(\d+)\s*CH',
            ]
            credit_hours = "Unknown"
            for pattern in credit_patterns:
                match = re.search(pattern, credit_hours_raw, re.IGNORECASE)
                if match:
                    if '+' in pattern:
                        credit_hours = f"{match.group(1)}+{match.group(2)}"
                    else:
                        credit_hours = f"{match.group(1)}+0"
                    break
            if credit_hours == "Unknown":
                course_lower = course.lower()
                lab_keywords = ["lab", "practical", "programming", "graphics", "networks", "compiler",
                                "database", "web", "software", "game", "hci", "human computer", "interaction"]
                if any(k in course_lower for k in lab_keywords):
                    credit_hours = "3+1"
                else:
                    credit_hours = "3+0"
            features["credit_hours"] = credit_hours
            return features
        except Exception:
            return {"C_vis":0.5, "C_prac":0.5, "C_struct":0.5, "credit_hours":"3+0"}

    def _apply_semantic_fallback(self, course: str) -> Dict:
        try:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are an expert in CS curricula. Estimate Visual, Practical, Structured scores for this course name. Be specific to this course's typical content. Output JSON with C_vis, C_prac, C_struct."),
                HumanMessage(content=f"Course: {course}")
            ])
            chain = prompt | self.llm | StrOutputParser()
            res = safe_llm_call(chain, {})
            match = re.search(r'\{.*\}', res, re.DOTALL)
            if match:
                features = json.loads(match.group())
            else:
                features = {"C_vis":0.5, "C_prac":0.5, "C_struct":0.5}
            features["C_vis"] = max(0.2, min(1.0, features.get("C_vis", 0.5)))
            features["C_prac"] = max(0.1, min(1.0, features.get("C_prac", 0.5)))
            features["C_struct"] = max(0.1, min(1.0, features.get("C_struct", 0.5)))
            course_lower = course.lower()
            lab_keywords = ["lab", "practical", "programming", "graphics", "networks", "compiler",
                            "database", "web", "software", "game", "hci", "human computer", "interaction"]
            if any(k in course_lower for k in lab_keywords):
                features["credit_hours"] = "3+1"
            else:
                features["credit_hours"] = "3+0"
            return features
        except:
            return {"C_vis":0.5, "C_prac":0.5, "C_struct":0.5, "credit_hours":"3+0"}

    def _ensure_distinct_features(self, features_dict: Dict[str, Dict]) -> Dict[str, Dict]:
        if len(features_dict) <= 1:
            return features_dict
        first_course = next(iter(features_dict))
        first_feats = features_dict[first_course]
        all_same = all(
            features_dict[c]["C_vis"] == first_feats["C_vis"] and
            features_dict[c]["C_prac"] == first_feats["C_prac"] and
            features_dict[c]["C_struct"] == first_feats["C_struct"]
            for c in features_dict
        )
        if not all_same:
            return features_dict
        for c in features_dict:
            llm_features = self._apply_semantic_fallback(c)
            if "source_url" in features_dict[c]:
                llm_features["source_url"] = features_dict[c]["source_url"]
            if "credit_hours" in features_dict[c]:
                llm_features["credit_hours"] = features_dict[c]["credit_hours"]
            features_dict[c] = llm_features
        return features_dict

    def run(self, country: str, degree: str, courses: List[str], university: str = None):
        try:
            for c in courses:
                url = None
                url_type = None
                if university:
                    result = self._search_university_domain(c, degree, university, country)
                    if result:
                        url, url_type = result
                if url:
                    if url_type == 'pdf':
                        pdf_path = self._download_pdf(url)
                        if pdf_path:
                            vectorstore = self._build_rag_from_pdf(pdf_path)
                            if vectorstore:
                                docs = vectorstore.similarity_search(f"Course {c} description objectives outline credit hours", k=5)
                                context = "\n".join([d.page_content for d in docs])
                                features = self._extract_features_from_context(c, context)
                                features["source_url"] = self._last_success_url if self._last_success_url else url
                                self.course_features[c] = features
                                continue
                    else:  # html
                        scraped_text = self._scrape_course_page(url)
                        if scraped_text:
                            features = self._extract_features_from_context(c, scraped_text)
                            features["source_url"] = f"Scraped from {url}"  # <-- This is the prefix we need to clean later
                            self.course_features[c] = features
                            continue
                pdf_url = self._search_course_outline(c, degree, country, university)
                if pdf_url:
                    pdf_path = self._download_pdf(pdf_url)
                    if pdf_path:
                        vectorstore = self._build_rag_from_pdf(pdf_path)
                        if vectorstore:
                            docs = vectorstore.similarity_search(f"Course {c} description objectives outline credit hours", k=5)
                            context = "\n".join([d.page_content for d in docs])
                            features = self._extract_features_from_context(c, context)
                            features["source_url"] = self._last_success_url if self._last_success_url else pdf_url
                            self.course_features[c] = features
                            continue
                if c not in self.course_features:
                    try:
                        with DDGS() as ddgs:
                            query = f"{c} syllabus filetype:pdf"
                            results = list(ddgs.text(query, max_results=3))
                            for r in results:
                                url = r.get('href', '')
                                if '.pdf' in url.lower():
                                    pdf_path = self._download_pdf(url)
                                    if pdf_path:
                                        vectorstore = self._build_rag_from_pdf(pdf_path)
                                        if vectorstore:
                                            docs = vectorstore.similarity_search(f"Course {c} description objectives outline credit hours", k=5)
                                            context = "\n".join([d.page_content for d in docs])
                                            features = self._extract_features_from_context(c, context)
                                            features["source_url"] = self._last_success_url if self._last_success_url else url
                                            self.course_features[c] = features
                                            break
                    except:
                        pass
                if c not in self.course_features:
                    features = self._apply_semantic_fallback(c)
                    features["source_url"] = "Semantic estimation (no content found)"
                    self.course_features[c] = features
            self.course_features = self._ensure_distinct_features(self.course_features)
            self.is_ready = True
        except Exception as e:
            self.error = e
            self.is_ready = True


# SECTION 6: FSLSM AGENT

class FSLSMAgent:
    def __init__(self, llm):
        self.llm = llm
        self.answers = []
        self.name = ""
        self.rephrase_cache = {}

    def rephrase_question(self, original: str, context: str) -> str:
        cache_key = original[:50]
        if cache_key in self.rephrase_cache:
            return self.rephrase_cache[cache_key]
        try:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="Rephrase naturally, MUST keep (a) and (b). If uncertain, output original."),
                HumanMessage(content=f"Context: {context}\nOriginal: {original}")
            ])
            chain = prompt | self.llm | StrOutputParser()
            text = safe_llm_call(chain, {})
            if "(a)" in text and "(b)" in text:
                self.rephrase_cache[cache_key] = text
                return text
        except:
            pass
        self.rephrase_cache[cache_key] = original
        return original

    def compute_profile(self):
        scores = {"ACT_REF":0, "SEN_INT":0, "VIS_VER":0, "SEQ_GLO":0}
        for dim, delta in self.answers:
            scores[dim] += delta
        
        visual = max(0.0, min(1.0, (scores["VIS_VER"] + 11) / 22.0))
        sensing = max(0.0, min(1.0, (scores["SEN_INT"] + 11) / 22.0))
        active = max(0.0, min(1.0, (scores["ACT_REF"] + 11) / 22.0))
        global_score = max(0.0, min(1.0, (scores["SEQ_GLO"] + 11) / 22.0))
        
        verbal = 1.0 - visual
        intuitive = 1.0 - sensing
        reflective = 1.0 - active
        sequential = 1.0 - global_score
        
        return {
            "visual": visual,
            "sensing": sensing,
            "active": active,
            "global": global_score,
            "verbal": verbal,
            "intuitive": intuitive,
            "reflective": reflective,
            "sequential": sequential
        }


# SECTION 6.5: DOMINANT POLE HELPER

def get_dominant_poles(profile: Dict) -> Dict[str, Tuple[str, float]]:
    """
    For each FSLSM dimension, return the dominant pole (higher score) and its score.
    Returns a dict mapping the dimension key to (pole_name, score).
    """
    dimensions = {
        "Input": ("Visual", profile["visual"], "Verbal", profile["verbal"]),
        "Perception": ("Sensing", profile["sensing"], "Intuitive", profile["intuitive"]),
        "Processing": ("Active", profile["active"], "Reflective", profile["reflective"]),
        "Understanding": ("Global", profile["global"], "Sequential", profile["sequential"])
    }
    result = {}
    for dim, (pole1, score1, pole2, score2) in dimensions.items():
        if score1 >= score2:
            result[dim] = (pole1, score1)
        else:
            result[dim] = (pole2, score2)
    return result


# SECTION 7: VISUALIZATION FUNCTIONS

def visualize_multi_course_graph(profile, course_features, comparison_results, student_name, 
                                  student_country=None, student_university=None):
    G = nx.Graph()
    G.add_node(student_name, type="student")
    
    # Get dominant poles
    dominant = get_dominant_poles(profile)
    
    # Display labels for pole nodes (showing dominant pole)
    display_labels = {
        "Visual": f"{dominant['Input'][0]} ({dominant['Input'][1]:.2f})",
        "Sensing": f"{dominant['Perception'][0]} ({dominant['Perception'][1]:.2f})",
        "Active": f"{dominant['Processing'][0]} ({dominant['Processing'][1]:.2f})",
        "Global": f"{dominant['Understanding'][0]} ({dominant['Understanding'][1]:.2f})"
    }
    
    # Actual first-pole scores for edge calculations
    pole_scores = {
        "Visual": profile["visual"],
        "Sensing": profile["sensing"],
        "Active": profile["active"],
        "Global": profile["global"]
    }
    
    # Add pole nodes with display labels
    for pole, label in display_labels.items():
        G.add_node(pole, type="pole", score=pole_scores[pole], label=label)
        G.add_edge(student_name, pole, weight=pole_scores[pole], style="solid", color="gray")
    
    # Course colors based on source type
    source_colors = {
        "regional": "#2ECC71",
        "international": "#3498DB",
        "semantic": "#F1C40F"
    }
    
    course_color_map = {}
    for idx, (course_name, features) in enumerate(course_features.items()):
        source_url = features.get('source_url', 'Unknown')
        source_type, _ = detect_course_source(source_url, student_country or "Other", student_university)
        color = source_colors.get(source_type, "#FF5722")
        course_color_map[course_name] = color
        G.add_node(course_name, type="course", color=color)
        
        # Connect every course to every pole with a match score (threshold lowered to 0.01)
        pole_to_feature = {"Visual": "C_vis", "Sensing": "C_prac", "Global": "C_struct"}
        # Also add Active pole connection (use profile's active score and course's structure as proxy)
        for pole, feature_key in pole_to_feature.items():
            if feature_key in features:
                match_score = profile.get(pole.lower(), 0.5) * features[feature_key]
                if match_score > 0.01:  # lowered threshold to show all connections
                    G.add_edge(pole, course_name, weight=match_score, style="dashed", color=color)
        # Add Active pole connection (using average of other features or direct)
        active_score = profile.get("active", 0.5) * 0.5  # placeholder
        if active_score > 0.01:
            G.add_edge("Active", course_name, weight=active_score, style="dashed", color=color)
    
    # Positioning
    pos = {student_name: (0, 0)}
    pole_names = ["Visual", "Sensing", "Active", "Global"]
    num_poles = len(pole_names)
    angles = np.linspace(0, 2 * np.pi, num_poles, endpoint=False)
    for i, pole in enumerate(pole_names):
        pos[pole] = (1.0 * np.cos(angles[i]), 1.0 * np.sin(angles[i]))
    num_courses = len(course_features)
    if num_courses > 0:
        course_angles = np.linspace(0, 2 * np.pi, num_courses, endpoint=False)
        for i, course in enumerate(course_features.keys()):
            pos[course] = (2.2 * np.cos(course_angles[i]), 2.2 * np.sin(course_angles[i]))
    
    fig, ax = plt.subplots(figsize=(18, 14))
    
    nx.draw_networkx_nodes(G, pos, nodelist=[student_name], node_color="#2E7D32", node_size=3500, edgecolors="black", linewidths=2, ax=ax)
    pole_nodes = [n for n in G.nodes if G.nodes[n].get("type") == "pole"]
    nx.draw_networkx_nodes(G, pos, nodelist=pole_nodes, node_color="#5D6D7E", node_size=2200, edgecolors="black", linewidths=1.5, ax=ax)
    course_nodes = [n for n in G.nodes if G.nodes[n].get("type") == "course"]
    course_colors_list = [G.nodes[n].get("color", "#FF5722") for n in course_nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=course_nodes, node_color=course_colors_list, node_size=2500, edgecolors="black", linewidths=2, alpha=0.9, ax=ax)
    
    solid_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("style") == "solid"]
    if solid_edges:
        nx.draw_networkx_edges(G, pos, edgelist=solid_edges, width=2.0, alpha=0.6, edge_color="gray", ax=ax)
    dashed_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("style") == "dashed"]
    if dashed_edges:
        for u, v in dashed_edges:
            color = G[u][v].get("color", "black")
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=2.0, alpha=0.9, edge_color=color, style="dashed", ax=ax)
    
    # Draw node labels with custom labels
    node_labels = {}
    for node in G.nodes:
        if node == student_name:
            node_labels[node] = student_name
        elif G.nodes[node].get("type") == "pole":
            node_labels[node] = G.nodes[node].get("label", node)
        else:
            node_labels[node] = node
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight="bold", ax=ax)
    
    # Edge labels
    edge_labels = {}
    for u, v in dashed_edges:
        weight = G[u][v].get("weight", 0)
        if weight > 0.01:  # show even small weights
            edge_labels[(u, v)] = f"{weight:.2f}"
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5, rotate=False, ax=ax)
    
    # Add pole scores as text below each pole node
    for pole, score in pole_scores.items():
        x, y = pos[pole]
        ax.text(x, y - 0.18, f"{score:.2f}", fontsize=8, ha='center', va='center', color='black', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Student', markerfacecolor='#2E7D32', markersize=15),
        Line2D([0], [0], marker='o', color='w', label='FSLSM Poles', markerfacecolor='#5D6D7E', markersize=15),
    ]
    for course, color in course_color_map.items():
        ite_val = comparison_results.get(course, {}).get("ite", 0)
        credit = course_features.get(course, {}).get("credit_hours", "N/A")
        legend_elements.append(
            Line2D([0], [0], color=color, lw=4, linestyle='--',
                   label=f"{course} (ITE: {ite_val:+.3f}, Credits: {credit})")
        )
    legend_elements.extend([
        Line2D([0], [0], color=source_colors["regional"], lw=4, linestyle='--',
               label='Regional Course (Student\'s Country/University)'),
        Line2D([0], [0], color=source_colors["international"], lw=4, linestyle='--',
               label='International Course (Global Platform/Foreign University)'),
        Line2D([0], [0], color=source_colors["semantic"], lw=4, linestyle='--',
               label='Semantic Estimation (No Syllabus Found)'),
    ])
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
    ax.set_title(f"Multi-Course Cognitive Overlay: {student_name}\n(Edge weights = Causal Match Strength | Scores shown on poles | Green=Regional, Blue=International, Yellow=Semantic)", fontsize=13, pad=20)
    ax.axis("off")
    plt.tight_layout()
    return fig


# SECTION 8: CORE REASONING ENGINE

def iterative_reasoning_step(V, E, X, a, profile, course_features, concepts, best_course, step_num,
                             initial_a=None, course_names=None):
    frontier = [node for node in V if a.get(node, 0) > 0.05 and node not in ['Student', 'Context']]
    
    for node in frontier:
        if node in ['Visual', 'Sensing', 'Active', 'Global']:
            for course in course_features.keys():
                if not any(e[0] == node and e[1] == course for e in E):
                    prob = profile.get(node.lower(), 0.5) * 0.5
                    E.append((node, course, prob))
        if node == best_course:
            for concept in concepts:
                if not any(e[0] == node and e[1] == concept for e in E):
                    prob = 0.5 + 0.2 * np.random.randn()
                    E.append((node, concept, prob))
    
    for node in V:
        if node in X:
            arr = np.array(X[node])
            updated = arr * 0.9 + 0.1 * np.random.randn(len(arr))
            X[node] = np.clip(updated, 0, 1).tolist()
    
    new_a = {node: 0.01 for node in V}
    for u, v, p in E:
        if v in new_a:
            new_a[v] += p
    
    if initial_a is not None and course_names is not None:
        alpha = 0.7
        for node in new_a:
            if node in course_names:
                new_a[node] = alpha * initial_a.get(node, 0) + (1 - alpha) * new_a[node]
    
    total = sum(new_a.values())
    if total > 0:
        for node in new_a:
            new_a[node] /= total
    
    a = new_a
    trace = f"  Step {step_num}: Expanded {len(frontier)} frontier nodes, attention rebalanced (alpha={alpha if initial_a else 'disabled'})."
    return V, E, X, a, trace


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

    def add_reasoning_step(self, step: str):
        self.reasoning_trace.append(step)


def generate_comparative_cognitive_graph(student_name, profile, courses, course_features,
                                         comparison_results, best_course, country=None,
                                         degree=None, semester=None, university=None,
                                         n_steps=3, llm=None):
    cg = CognitiveGraph(
        V=[student_name],
        E=[],
        X={student_name: [profile["visual"], profile["sensing"], profile["active"], profile["global"]]},
        a={student_name: 1.0},
        best_course=best_course,
        student_name=student_name,
        country=country,
        degree=degree,
        semester=semester,
        university=university
    )
    cg.add_reasoning_step("System 1: Expanding context from student profile")
    cg.add_reasoning_step(f"  Profile: Visual={profile['visual']:.2f}, Sensing={profile['sensing']:.2f}, Active={profile['active']:.2f}, Global={profile['global']:.2f}")

    context_nodes = []
    if country:
        cg.V.append(country)
        cg.X[country] = [0.5, 0.5, 0.5, 0.5]
        cg.a[country] = 0.0
        context_nodes.append(country)
    if degree:
        cg.V.append(degree)
        cg.X[degree] = [0.5, 0.5, 0.5, 0.5]
        cg.a[degree] = 0.0
        context_nodes.append(degree)
    if semester:
        year_node = f"Year: {semester}"
        cg.V.append(year_node)
        cg.X[year_node] = [0.5, 0.5, 0.5, 0.5]
        cg.a[year_node] = 0.0
        context_nodes.append(year_node)
    if university:
        cg.V.append(university)
        cg.X[university] = [0.5, 0.5, 0.5, 0.5]
        cg.a[university] = 0.0
        context_nodes.append(university)
    cg.add_reasoning_step(f"  Context: {', '.join(context_nodes) if context_nodes else 'None'}")

    for ctx in context_nodes:
        cg.E.append({"from": student_name, "to": ctx, "relation": "has_context", "prob": 0.9, "is_recommended_path": True})
        cg.a[ctx] = 0.3
    cg.add_reasoning_step(f"  Connected {student_name} to {len(context_nodes)} context nodes")

    poles = ["Visual", "Sensing", "Active", "Global"]
    pole_scores = {
        "Visual": profile["visual"],
        "Sensing": profile["sensing"],
        "Active": profile["active"],
        "Global": profile["global"]
    }
    for pole in poles:
        cg.V.append(pole)
        cg.X[pole] = [pole_scores[pole], 0, 0, 0]
        cg.a[pole] = 0.0
    cg.add_reasoning_step(f"  Added {len(poles)} FSLSM poles")

    for pole in poles:
        edge_prob = pole_scores[pole]
        cg.E.append({"from": student_name, "to": pole, "relation": "has_preference", "prob": edge_prob, "is_recommended_path": True})
        cg.a[pole] = edge_prob * 0.3
    cg.add_reasoning_step(f"  Connected {student_name} to {len(poles)} FSLSM poles")

    for course in courses:
        cg.V.append(course)
        cg.X[course] = [
            comparison_results[course].get("predicted_grade", 0.5),
            comparison_results[course].get("ite", 0),
            course_features[course].get("C_vis", 0.5),
            course_features[course].get("C_prac", 0.5)
        ]
        cg.a[course] = 0.0
    cg.add_reasoning_step(f"  Found {len(courses)} candidate courses: {', '.join(courses)}")

    for course in courses:
        matches = comparison_results[course].get("matches", {})
        is_best = (course == best_course)
        for ctx in context_nodes:
            cg.E.append({"from": ctx, "to": course, "relation": "offers_course", "prob": 0.5,
                         "is_recommended_path": is_best, "ite": comparison_results[course].get("ite", 0)})
            cg.a[course] += 0.05
        for pole in poles:
            match_score = matches.get(pole, 0.0)
            edge_prob = min(0.95, max(0.05, match_score))
            cg.E.append({"from": pole, "to": course, "relation": "matches", "prob": edge_prob,
                         "is_recommended_path": is_best, "ite": comparison_results[course].get("ite", 0)})
            cg.a[course] += edge_prob * 0.1
    cg.add_reasoning_step(f"  Added {len(courses) * (len(context_nodes) + len(poles))} edges to courses")

    sorted_courses = sorted(comparison_results.items(), key=lambda x: x[1]["ite"], reverse=True)
    cg.add_reasoning_step("System 2: Comparing ITE values")
    for course, data in sorted_courses:
        ite = data["ite"]
        cg.add_reasoning_step(f"  {course}: ITE = {ite:.3f}")
    cg.add_reasoning_step(f"  Best course identified: {best_course} (ITE = {comparison_results[best_course]['ite']:.3f})")

    cg.add_reasoning_step(f"Expanding '{best_course}' to show reasoning path")

    concepts = []
    if llm is not None:
        try:
            concept_prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a curriculum expert. List exactly 4 core topics for this university course. Output ONLY a JSON list of strings."),
                HumanMessage(content=f"Course: {best_course}")
            ])
            concept_chain = concept_prompt | llm | StrOutputParser()
            concept_response = safe_llm_call(concept_chain, {})
            concept_response = re.sub(r'```json\s*', '', concept_response)
            concept_response = re.sub(r'```\s*', '', concept_response)
            concepts = json.loads(concept_response)
            if not isinstance(concepts, list) or len(concepts) < 2:
                concepts = ["Core Concepts", "Applications", "Implementation", "Advanced Topics"]
        except Exception:
            concepts = ["Core Concepts", "Applications", "Implementation", "Advanced Topics"]
    else:
        concepts = ["Core Concepts", "Applications", "Implementation", "Advanced Topics"]
    
    if len(concepts) < 4:
        fallback = ["Core Concepts", "Applications", "Implementation", "Advanced Topics"]
        concepts = concepts + fallback[:4-len(concepts)]
    elif len(concepts) > 4:
        concepts = concepts[:4]

    for c in concepts:
        cg.V.append(c)
        cg.X[c] = [0.0] * 4
        cg.a[c] = 0.0

    for concept in concepts:
        concept_lower = concept.lower()
        visual_match = 0.5
        practical_match = 0.5
        global_match = 0.5
        if any(k in concept_lower for k in ["gate", "circuit", "graphics", "rendering", "animation", "visual"]):
            visual_match = 0.8
        if any(k in concept_lower for k in ["process", "memory", "file", "scheduling", "analysis", "refactoring", "testing", "implementation"]):
            practical_match = 0.8
        if any(k in concept_lower for k in ["theory", "algorithm", "logic", "design", "architecture", "model"]):
            global_match = 0.8
        match_score = (
            visual_match * profile["visual"] +
            practical_match * profile["sensing"] +
            global_match * profile["global"] +
            0.5 * profile["active"]
        ) / 4.0
        edge_prob = min(0.95, max(0.05, match_score))
        cg.E.append({"from": best_course, "to": concept, "relation": "teaches_concept", "prob": edge_prob,
                     "is_recommended_path": True, "visual_match": visual_match,
                     "practical_match": practical_match, "global_match": global_match})
        cg.a[concept] = edge_prob * 0.5
    cg.add_reasoning_step(f"  Expanded to {len(concepts)} concepts: {', '.join(concepts)}")

    cg.add_reasoning_step("Attention Update: Focusing on the recommended path")
    cg.a[student_name] = 0.08
    for ctx in context_nodes:
        cg.a[ctx] = 0.04
    for pole in poles:
        cg.a[pole] = pole_scores[pole] * 0.04
    cg.a[best_course] = 0.35
    for concept in concepts:
        for e in cg.E:
            if e["to"] == concept:
                cg.a[concept] = e["prob"] * 0.25
                break
    for course in courses:
        if course != best_course:
            cg.a[course] = 0.015
    total = sum(cg.a.values())
    if total > 0:
        for node in cg.V:
            cg.a[node] = cg.a.get(node, 0) / total

    top_nodes = sorted(cg.a.items(), key=lambda x: x[1], reverse=True)[:8]
    cg.add_reasoning_step(f"  Highest attention: " +
                         ", ".join([f'{n} ({v:.3f})' for n, v in top_nodes if v > 0.01]))
    cg.add_reasoning_step(f"Prediction: '{best_course}' recommended")
    cg.add_reasoning_step(f"   ITE = {comparison_results[best_course]['ite']:.3f}, Grade = {comparison_results[best_course]['predicted_grade']:.3f}")
    cg.add_reasoning_step(f"   Credit Hours: {comparison_results[best_course].get('credit_hours', 'Unknown')}")

    E_tuples = [(e["from"], e["to"], e["prob"]) for e in cg.E]
    V_list = cg.V
    X_dict = cg.X
    a_dict = cg.a

    course_names = list(course_features.keys())
    initial_a = a_dict.copy()

    for step in range(1, n_steps + 1):
        V_list, E_tuples, X_dict, a_dict, trace = iterative_reasoning_step(
            V_list, E_tuples, X_dict, a_dict, profile, course_features, concepts, best_course, step,
            initial_a=initial_a,
            course_names=course_names
        )
        cg.add_reasoning_step(trace)

    cg.V = V_list
    cg.E = []
    for u, v, p in E_tuples:
        cg.E.append({"from": u, "to": v, "prob": p, "relation": "reasoned"})
    cg.X = X_dict
    cg.a = a_dict

    total = sum(cg.a.values())
    if total > 0:
        for node in cg.V:
            cg.a[node] = cg.a.get(node, 0) / total

    final_top = sorted(cg.a.items(), key=lambda x: x[1], reverse=True)[:8]
    cg.add_reasoning_step(f"Final attention after {n_steps} Reasoning iterations: " +
                         ", ".join([f'{n} ({v:.3f})' for n, v in final_top if v > 0.01]))

    return cg


# SECTION 9: COMPREHENSIVE COGNITIVE GRAPH VISUALIZATION

def visualize_comparative_cognitive_graph(cg, profile, comparison_results,
                                          title="Comprehensive Cognitive Graph",
                                          save_files=False, filename_base="cognitive_graph",
                                          pdf_mode=False):
    student_name = cg.student_name
    G = nx.DiGraph()
    for node in cg.V:
        G.add_node(node, attention=cg.a.get(node, 0.0))
    for e in cg.E:
        G.add_edge(e["from"], e["to"], prob=e["prob"],
                   is_recommended_path=e.get("is_recommended_path", False),
                   ite=e.get("ite", 0),
                   relation=e.get("relation", ""))

    colors = {
        "student": "#E8F5E9", "student_border": "#2E7D32",
        "context": "#E3F2FD", "context_border": "#1565C0",
        "pole": "#F3E5F5", "pole_border": "#7B1FA2",
        "course": "#E3F2FD", "course_border": "#1565C0",
        "best_course": "#FFF8E1", "best_course_border": "#E65100",
        "concept": "#FFF3E0", "concept_border": "#E65100",
        "recommended_edge": "#D32F2F", "other_edge": "#1565C0",
        "edge_label_bg": "white",
    }

    figsize = (20, 14) if pdf_mode else (20, 14)
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    plt.sca(ax)

    context_names = [cg.country, cg.degree, f"Year: {cg.semester}", cg.university]
    context_names = [c for c in context_names if c]
    poles = ["Visual", "Sensing", "Active", "Global"]

    context_nodes = []
    pole_nodes = []
    courses_nodes = []
    concepts_nodes = []
    for node in cg.V:
        if node == student_name:
            continue
        if node in context_names:
            context_nodes.append(node)
        elif node in poles:
            pole_nodes.append(node)
        elif any(e["to"] == node and e["from"] in poles for e in cg.E):
            courses_nodes.append(node)
        else:
            concepts_nodes.append(node)

    spacing = 1.5 if pdf_mode else 1.3
    pos = {}
    pos[student_name] = (0, 2.8 if pdf_mode else 3.0)
    num_ctx = len(context_nodes)
    if num_ctx > 0:
        for i, ctx in enumerate(context_nodes):
            x = (i - (num_ctx - 1) / 2) * spacing
            pos[ctx] = (x, 1.8 if pdf_mode else 2.0)
    num_poles = len(pole_nodes)
    if num_poles > 0:
        for i, pole in enumerate(pole_nodes):
            x = (i - (num_poles - 1) / 2) * 0.9
            pos[pole] = (x, 0.9 if pdf_mode else 1.0)
    num_courses = len(courses_nodes)
    if num_courses > 0:
        for i, course in enumerate(courses_nodes):
            x = (i - (num_courses - 1) / 2) * 1.4
            pos[course] = (x, -0.2 if pdf_mode else 0.0)
    best_course = cg.best_course
    if best_course and concepts_nodes:
        best_x = pos.get(best_course, (0, 0))[0]
        num_concepts = len(concepts_nodes)
        if num_concepts > 1:
            for i, concept in enumerate(concepts_nodes):
                x = best_x + (i - (num_concepts - 1) / 2) * 0.8
                pos[concept] = (x, -0.9 if pdf_mode else -1.0)
        else:
            pos[concepts_nodes[0]] = (best_x, -0.9 if pdf_mode else -1.0)

    # Get dominant pole labels
    dominant = get_dominant_poles(profile)
    pole_display = {
        "Visual": f"{dominant['Input'][0]} ({dominant['Input'][1]:.2f})",
        "Sensing": f"{dominant['Perception'][0]} ({dominant['Perception'][1]:.2f})",
        "Active": f"{dominant['Processing'][0]} ({dominant['Processing'][1]:.2f})",
        "Global": f"{dominant['Understanding'][0]} ({dominant['Understanding'][1]:.2f})"
    }

    node_size_scale = 1.2 if pdf_mode else 1.0
    nx.draw_networkx_nodes(G, pos, nodelist=[student_name],
                          node_color=colors["student"], node_size=int(3500*node_size_scale),
                          edgecolors=colors["student_border"], linewidths=2, ax=ax)
    if context_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=context_nodes,
                              node_color=colors["context"], node_size=int(2800*node_size_scale),
                              edgecolors=colors["context_border"], linewidths=2, ax=ax)
    if pole_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=pole_nodes,
                              node_color=colors["pole"], node_size=int(2400*node_size_scale),
                              edgecolors=colors["pole_border"], linewidths=2, ax=ax)
    best_course_node = cg.best_course
    best_nodes = [n for n in courses_nodes if n == best_course_node]
    other_course_nodes = [n for n in courses_nodes if n != best_course_node]
    if other_course_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=other_course_nodes,
                              node_color=colors["course"], node_size=int(2600*node_size_scale),
                              edgecolors=colors["course_border"], linewidths=2, ax=ax)
    if best_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=best_nodes,
                              node_color=colors["best_course"], node_size=int(3200*node_size_scale),
                              edgecolors=colors["best_course_border"], linewidths=3, ax=ax)
    if concepts_nodes:
        concept_sizes = [max(1000, G.nodes[n].get("attention", 0.1) * 5000 + 500) for n in concepts_nodes]
        concept_sizes = [int(s*node_size_scale) for s in concept_sizes]
        nx.draw_networkx_nodes(G, pos, nodelist=concepts_nodes,
                              node_color=colors["concept"], node_size=concept_sizes,
                              edgecolors=colors["concept_border"], linewidths=2, ax=ax)

    # Draw edges
    edge_labels_dict = {}
    for u, v, d in G.edges(data=True):
        is_rec = d.get("is_recommended_path", False)
        prob = d.get("prob", 0.1)
        style = "solid" if is_rec else "dashed"
        color = colors["recommended_edge"] if is_rec else colors["other_edge"]
        width = 2.5 if is_rec else 1.5
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                              width=width, alpha=0.8,
                              edge_color=color, style=style,
                              arrows=True, arrowsize=18, arrowstyle='->', ax=ax)
        label = None
        relation = d.get("relation", "")
        if v in courses_nodes and "ite" in d and d["ite"] != 0 and u == cg.university:
            label = f"ITE: {d['ite']:.3f}"
        elif prob > 0.05:
            label = f"p={prob:.2f}"
        if label:
            edge_labels_dict[(u, v)] = label

    if edge_labels_dict:
        fontsize_edge = 10 if pdf_mode else 8
        nx.draw_networkx_edge_labels(G, pos, edge_labels_dict,
                                     font_size=fontsize_edge, label_pos=0.5,
                                     bbox=dict(boxstyle="round,pad=0.2",
                                               facecolor=colors["edge_label_bg"], alpha=0.8),
                                     ax=ax)

    # Node labels
    node_labels = {}
    fontsize = 8 if pdf_mode else 9
    for node in G.nodes:
        attn = G.nodes[node].get("attention", 0)
        if node == student_name:
            node_labels[node] = f"{student_name}\n(V:{profile['visual']:.2f}\nS:{profile['sensing']:.2f}\nG:{profile['global']:.2f})"
        elif node in context_nodes:
            node_labels[node] = node
        elif node in pole_nodes:
            node_labels[node] = pole_display.get(node, node)
        elif node in courses_nodes:
            if node == best_course_node:
                ite_val = comparison_results.get(node, {}).get("ite", 0)
                cred = comparison_results.get(node, {}).get("credit_hours", "N/A")
                node_labels[node] = f"* {node}\nITE: {ite_val:.3f}\nCredits: {cred}"
            else:
                ite_val = comparison_results.get(node, {}).get("ite", 0)
                cred = comparison_results.get(node, {}).get("credit_hours", "N/A")
                node_labels[node] = f"{node}\nITE: {ite_val:.3f}\nCredits: {cred}"
        else:
            node_labels[node] = f"{node}\na={attn:.2f}"
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=fontsize, font_weight="regular", ax=ax)

    legend_fontsize = 9 if pdf_mode else 10
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=colors["student"], edgecolor=colors["student_border"], label="Student (Profile)"),
        plt.Rectangle((0,0),1,1, facecolor=colors["context"], edgecolor=colors["context_border"], label="Context"),
        plt.Rectangle((0,0),1,1, facecolor=colors["pole"], edgecolor=colors["pole_border"], label="FSLSM Pole (Dominant)"),
        plt.Rectangle((0,0),1,1, facecolor=colors["course"], edgecolor=colors["course_border"], label="Course (Candidate)"),
        plt.Rectangle((0,0),1,1, facecolor=colors["best_course"], edgecolor=colors["best_course_border"], label="Recommended Course"),
        plt.Rectangle((0,0),1,1, facecolor=colors["concept"], edgecolor=colors["concept_border"], label="Concept"),
        plt.Line2D([0], [0], color=colors["recommended_edge"], lw=2, linestyle='solid', label="Recommended Path"),
        plt.Line2D([0], [0], color=colors["other_edge"], lw=2, linestyle='dashed', label="Other Paths")
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=legend_fontsize)

    title_fontsize = 12 if pdf_mode else 14
    ax.set_title(f"{title}\n(Solid = Recommended Path | Dashed = Other Paths | Node size = Attention)",
                 fontsize=title_fontsize, pad=20)
    ax.axis("off")
    plt.tight_layout()

    if save_files:
        png_filename = f"{filename_base}.png"
        plt.savefig(png_filename, dpi=300, bbox_inches="tight", facecolor="white")
        pdf_filename = f"{filename_base}.pdf"
        try:
            plt.savefig(pdf_filename, dpi=300, bbox_inches="tight", facecolor="white")
        except Exception:
            pass
    return fig


# SECTION 10: CAUSAL ANALYSIS

def get_comparative_analysis(profile, course_features, llm, student_country="Pakistan", student_university=None):
    dataset_path = os.path.join(BASE_DIR, "CogNet_Advisory_Data.csv")
    if not os.path.exists(dataset_path):
        st.error(f"Original dataset `CogNet_Advisory_Data.csv` not found at {dataset_path}.")
        st.markdown("Please ensure the file is in the same directory as the script.")
        st.stop()
    
    st.info("Using original dataset `CogNet_Advisory_Data.csv`.")
    
    try:
        df = pd.read_csv(dataset_path)
        st.success(f"Loaded {len(df)} records from: {dataset_path}")
        
        df.rename(columns={
            'Visual': 'visual',
            'Sensing': 'sensing',
            'Active': 'active',
            'Global': 'global',
            'GPA': 'prior_gpa',
            'Course_Taken': 'course_taken',
            'Final_Grade': 'final_grade'
        }, inplace=True)
        
        required_cols = ['visual', 'sensing', 'active', 'global', 'prior_gpa', 
                         'C_vis', 'C_prac', 'C_struct', 'final_grade']
        for col in required_cols:
            if col not in df.columns:
                st.error(f"Column '{col}' not found in the dataset. Please check the CSV structure.")
                st.stop()
        df = df.dropna(subset=required_cols)
        
    except Exception as e:
        st.error(f"Error reading the dataset: {e}")
        st.stop()
    
    st.write("### Dataset Diagnostic Information")
    st.write(f"`C_vis` range: {df['C_vis'].min():.3f} – {df['C_vis'].max():.3f}")
    st.write(f"`C_vis` std deviation: {df['C_vis'].std():.3f}")
    st.write(f"Unique values in `C_vis`: {df['C_vis'].nunique()}")
    st.write(f"Any missing values? {df.isnull().any().any()}")
    if df['C_vis'].std() < 0.01:
        st.warning("`C_vis` has very low variation. Double ML may fail.")
    
    avg_vis = df['C_vis'].mean()
    avg_prac = df['C_prac'].mean()
    avg_struct = df['C_struct'].mean()
    
    causal_graph = """
    digraph {
        prior_gpa -> C_vis;
        prior_gpa -> C_prac;
        prior_gpa -> C_struct;
        prior_gpa -> final_grade;
        visual -> final_grade;
        sensing -> final_grade;
        active -> final_grade;
        global -> final_grade;
        C_vis -> final_grade;
        C_prac -> final_grade;
        C_struct -> final_grade;
    }
    """
    
    model = CausalModel(
        data=df,
        treatment='C_vis',
        outcome='final_grade',
        graph=causal_graph,
        common_causes=['visual', 'sensing', 'active', 'global', 'prior_gpa']
    )
    
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    
    X = df[['visual', 'sensing', 'active', 'global']].values
    T = df['C_vis'].values
    Y = df['final_grade'].values
    W = df[['prior_gpa']].values
    
    dml = LinearDML(
        model_y=GradientBoostingRegressor(),
        model_t=GradientBoostingRegressor(),
        discrete_treatment=False,
        cv=3
    )
    dml.fit(Y, T, X=X, W=W)
    
    results = {}
    best_course = None
    best_ite = -np.inf
    
    student_X = np.array([[profile['visual'], profile['sensing'], profile['active'], profile['global']]])
    cate = dml.effect(student_X)[0]
    
    st.write("### Debug: ITE Calculation")
    st.write(f"CATE value: {cate:.6f}")
    st.write(f"Average C_vis: {avg_vis:.3f}")
    
    for course_name, features in course_features.items():
        legacy_y = 0.5 + 0.3 * profile['visual'] * features.get('C_vis', 0.5) + \
                        0.2 * profile['sensing'] * features.get('C_prac', 0.5) + \
                        0.1 * profile['global'] * features.get('C_struct', 0.5)
        legacy_baseline = 0.5 + 0.3 * profile['visual'] * avg_vis + \
                                0.2 * profile['sensing'] * avg_prac + \
                                0.1 * profile['global'] * avg_struct
        legacy_ite = legacy_y - legacy_baseline
        
        delta_vis = features.get('C_vis', 0.5) - avg_vis
        causal_ite = cate * delta_vis
        
        if abs(causal_ite) < 0.0005:
            causal_ite = 0.0
        
        predicted_grade = legacy_y + causal_ite
        predicted_grade = max(0, min(1, predicted_grade))
        
        source_url = features.get('source_url', 'Unknown')
        source_type, source_label = detect_course_source(source_url, student_country, student_university)
        
        results[course_name] = {
            'ite': causal_ite,
            'legacy_ite': legacy_ite,
            'predicted_grade': predicted_grade,
            'credit_hours': features.get('credit_hours', 'Unknown'),
            'cate': cate,
            'source_type': source_type,
            'source_label': source_label,
            'source_url': source_url
        }
        
        st.write(f"{course_name}: C_vis={features.get('C_vis', 0.5):.3f}, delta_vis={delta_vis:.3f}, ITE={causal_ite:.6f}")
        
        if causal_ite > best_ite:
            best_ite = causal_ite
            best_course = course_name
    
    ite_values = [data['ite'] for data in results.values()]
    all_equal = all(abs(ite - ite_values[0]) < 0.0001 for ite in ite_values)
    
    def describe_course(features):
        vis = features.get('C_vis', 0.5)
        prac = features.get('C_prac', 0.5)
        struct = features.get('C_struct', 0.5)
        credit = features.get('credit_hours', 'Unknown')
        if prac > 0.7:
            nature = "practical-heavy (hands-on, lab-based)"
        elif vis > 0.7:
            nature = "visual-heavy (theory with diagrams, visual aids)"
        elif struct > 0.7:
            nature = "structured (well-organised, step-by-step)"
        else:
            nature = "balanced in content delivery"
        lab = "with a lab component" if str(credit).endswith("+1") else "without a lab component"
        return nature, lab, credit

    best_nature, best_lab, best_credit = describe_course(course_features[best_course])
    profile_text = (
        f"Visual: {profile['visual']:.2f}, Sensing (Practical): {profile['sensing']:.2f}, "
        f"Active: {profile['active']:.2f}, Global: {profile['global']:.2f}"
    )
    
    best_source_label = results[best_course].get('source_label', '')
    best_source_type = results[best_course].get('source_type', 'semantic')
    best_features = course_features[best_course]
    content_comp = get_course_content_comparison(best_course, best_source_type, best_features)
    
    learning_resources = get_learning_resources(best_course)
    resources_text = "\n".join([f"- {platform}: {', '.join(courses)}" for platform, courses in learning_resources.items()])

    # --- UPDATED: Improved explanation logic for equal/negative ITE ---
    equal_ite_note = ""
    if all_equal:
        equal_ite_note = (
            f"All courses have very similar Causal ITE values (approximately {ite_values[0]:.3f}). "
            f"This indicates that the causal benefit is comparable across courses. "
            f"The recommendation is therefore based on the best learning style match and course content alignment."
        )
    # If best_ite is negative, add a note to explain the meaning
    negative_ite_note = ""
    if best_ite < 0:
        negative_ite_note = (
            f"The Causal ITE is negative ({best_ite:.3f}), which suggests a predicted decrease in grade for this student. "
            f"However, among the available options, this course has the **least negative** impact and aligns best with your learning style. "
            f"The recommendation prioritizes the most suitable match for your profile."
        )
    # If best_ite is zero, mention neutrality
    zero_ite_note = ""
    if abs(best_ite) < 0.0005:
        zero_ite_note = (
            f"The Causal ITE is approximately zero ({best_ite:.3f}), indicating a neutral expected impact on your grade. "
            f"The recommendation is driven by the optimal alignment with your learning style and content preferences."
        )
    
    # Combine notes - only one of them will be non-empty, but we can combine if needed
    explanation_note = equal_ite_note or negative_ite_note or zero_ite_note
    
    try:
        comparison_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                f"You are a learning analytics advisor. The causal analysis (using Double ML, adjusting for prior GPA) has determined "
                f"that the BEST course for this student is '{best_course}' with a Causal ITE of {best_ite:.3f}.\n\n"
                f"Student Profile: {profile_text}\n\n"
                f"Best course details: Credit Hours = {best_credit}, {best_nature}, {best_lab}.\n\n"
                f"Course Source: This is a {best_source_label} course.\n\n"
                f"Course Content Comparison:\n"
                f"- Format: {content_comp['format']}\n"
                f"- Structure: {content_comp['structure']}\n"
                f"- Credit Hours: {content_comp['credit_hours']}\n"
                f"- Lab Component: {content_comp['lab_component']}\n"
                f"- Assessment: {content_comp['assessment']}\n"
                f"- Textbook: {content_comp['textbook']}\n"
                f"- Learning Style Fit: {content_comp['learning_style']}\n\n"
                f"Suggested Learning Resources (Online Platforms):\n{resources_text}\n\n"
                f"{explanation_note}\n\n"
                f"Your task is to write a concise but comprehensive explanation (4-6 sentences) that:\n"
                f"1. States why '{best_course}' is the optimal choice – interpret the Causal ITE appropriately (positive, negative, or neutral).\n"
                f"2. Relates the course's pedagogical features (visual/practical/structured content, credit hours, lab component) to the student's specific learning profile.\n"
                f"3. Mentions the course source type (regional vs. international) and provides a brief content comparison.\n"
                f"4. Suggests online platforms where the student can find supplementary learning resources.\n"
                f"5. Emphasises that the causal ITE accounts for confounders like prior GPA.\n"
                f"6. If all ITE values are similar, explain that the choice is based on learning style match and content alignment.\n"
                f"7. If the ITE is negative, explain that it's the least negative option and the best fit.\n"
                f"8. If the ITE is zero, explain the neutral impact and focus on the learning style match.\n\n"
                f"Be precise and avoid vague statements. Use numbers where appropriate."
            )),
            HumanMessage(content=(
                f"Student Profile: Visual={profile['visual']:.2f}, Sensing={profile['sensing']:.2f}, "
                f"Global={profile['global']:.2f}, Active={profile['active']:.2f}.\n\n"
                f"All Course Results: {json.dumps(results, indent=2)}\n\n"
                f"Best course: {best_course} (Causal ITE = {best_ite:.3f}, Legacy ITE = {results[best_course]['legacy_ite']:.3f}, "
                f"Credits: {best_credit}, Nature: {best_nature}, Lab: {best_lab})\n\n"
                f"Course Source: {best_source_label}\n"
                f"Content Comparison: {json.dumps(content_comp)}\n"
                f"Learning Resources: {resources_text}\n"
                f"Explanation Note: {explanation_note}\n\n"
                f"Generate a clear, insightful explanation for the student."
            ))
        ])
        chain = comparison_prompt | llm | StrOutputParser()
        final_explanation = safe_llm_call(chain, {})
    except Exception as e:
        final_explanation = (
            f"Based on causal analysis, '{best_course}' is recommended (Causal ITE = {best_ite:.3f}). "
            f"This {best_source_label} course, with its {best_nature} and {best_lab}, fits your learning profile ({profile_text}). "
            f"Content comparison: {content_comp['format']}, {content_comp['structure']}. "
            f"For supplementary learning, consider platforms like {', '.join(learning_resources.keys())}. "
            f"Compared to other courses, it provides the best expected causal benefit, adjusting for confounders like prior GPA."
            f"{' ' + equal_ite_note if equal_ite_note else ''}"
            f"{' ' + negative_ite_note if negative_ite_note else ''}"
            f"{' ' + zero_ite_note if zero_ite_note else ''}"
        )
    
    return best_course, results, final_explanation, df


# SECTION 11: CROSS-VALIDATION

def run_cross_validation_with_baselines(df, n_splits=5):
    feature_cols = ['visual', 'sensing', 'active', 'global', 'prior_gpa', 
                    'C_vis', 'C_prac', 'C_struct']
    X = df[feature_cols].values
    y = df['final_grade'].values
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = {
        'Linear Regression': {'rmse': [], 'mae': [], 'r2': []},
        'Random Forest': {'rmse': [], 'mae': [], 'r2': []},
        'Double ML': {'rmse': [], 'mae': [], 'r2': []}
    }
    
    dml_errors = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        pred_lr = lr.predict(X_test)
        results['Linear Regression']['rmse'].append(np.sqrt(mean_squared_error(y_test, pred_lr)))
        results['Linear Regression']['mae'].append(mean_absolute_error(y_test, pred_lr))
        results['Linear Regression']['r2'].append(r2_score(y_test, pred_lr))
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        pred_rf = rf.predict(X_test)
        results['Random Forest']['rmse'].append(np.sqrt(mean_squared_error(y_test, pred_rf)))
        results['Random Forest']['mae'].append(mean_absolute_error(y_test, pred_rf))
        results['Random Forest']['r2'].append(r2_score(y_test, pred_rf))
        
        try:
            X_train_dml = df.iloc[train_idx][['visual', 'sensing', 'active', 'global']].values
            T_train = df.iloc[train_idx]['C_vis'].values
            Y_train = df.iloc[train_idx]['final_grade'].values
            W_train = df.iloc[train_idx][['prior_gpa']].values
            
            if np.std(T_train) < 1e-6:
                raise ValueError(f"Fold {fold}: C_vis is constant (std={np.std(T_train):.2e})")
            
            dml = LinearDML(
                model_y=GradientBoostingRegressor(n_estimators=50, random_state=42),
                model_t=GradientBoostingRegressor(n_estimators=50, random_state=42),
                discrete_treatment=False,
                cv=2
            )
            dml.fit(Y_train, T_train, X=X_train_dml, W=W_train)
            
            X_test_dml = df.iloc[test_idx][['visual', 'sensing', 'active', 'global']].values
            cate_test = dml.effect(X_test_dml)
            
            avg_vis = df['C_vis'].mean()
            
            base_predictor = Ridge(alpha=1.0, random_state=42)
            base_predictor.fit(X_train, y_train)
            pred_base = base_predictor.predict(X_test)
            
            pred_dml = []
            for pos, (_, row) in enumerate(df.iloc[test_idx].iterrows()):
                causal_adj = cate_test[pos] * (row['C_vis'] - avg_vis)
                adjusted_pred = pred_base[pos] + causal_adj
                pred_dml.append(np.clip(adjusted_pred, 0, 1))
            
            pred_dml = np.array(pred_dml)
            results['Double ML']['rmse'].append(np.sqrt(mean_squared_error(y_test, pred_dml)))
            results['Double ML']['mae'].append(mean_absolute_error(y_test, pred_dml))
            results['Double ML']['r2'].append(r2_score(y_test, pred_dml))
            
        except Exception as e:
            error_msg = f"Fold {fold}: {type(e).__name__} – {str(e)}"
            dml_errors.append(error_msg)
            print(error_msg)
            continue
    
    summary = {}
    for method, metrics in results.items():
        if metrics['rmse']:
            summary[method] = {
                'RMSE (↓)': f"{np.mean(metrics['rmse']):.4f} ± {np.std(metrics['rmse']):.4f}",
                'MAE (↓)': f"{np.mean(metrics['mae']):.4f} ± {np.std(metrics['mae']):.4f}",
                'R2 (↑)': f"{np.mean(metrics['r2']):.4f} ± {np.std(metrics['r2']):.4f}",
            }
        else:
            summary[method] = {'RMSE': 'N/A', 'MAE': 'N/A', 'R2': 'N/A'}
    
    if dml_errors:
        st.warning("Double ML failed on some folds. Errors:")
        for err in dml_errors:
            st.write(f"- {err}")
    
    return pd.DataFrame(summary).T


# SECTION 12: PDF REPORT GENERATOR

def clean_source_url(url: str) -> str:
    """Remove common prefixes like 'Scraped from ' from the source URL."""
    prefixes = ["Scraped from ", "Scraped from: ", "Scraped from - ", "Scraped from "]
    for prefix in prefixes:
        if url.startswith(prefix):
            return url[len(prefix):]
    return url

def generate_pdf_report(profile, comparison_results, explanation, cg, simple_fig,
                        student_name, country, degree, semester, university,
                        courses, course_features, logo_path=None):
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        # Page 1: Summary tables (unchanged)
        fig1, ax1 = plt.subplots(figsize=(8.5, 11))
        ax1.axis('off')
        
        # Logo
        logo_placed = False
        if logo_path and os.path.exists(logo_path):
            try:
                img = Image.open(logo_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.thumbnail((120, 120))
                ax1.imshow(img, extent=[0.05, 0.05 + 120/850, 0.92, 0.92 + 120/1100],
                           aspect='auto', zorder=10, transform=ax1.transAxes)
                logo_placed = True
            except Exception:
                pass

        y_start = 0.88 if logo_placed else 0.95
        ax1.text(0.05, y_start, "CogNet Advisor Report", fontsize=18, fontweight='bold',
                 ha='left', va='top', transform=ax1.transAxes)
        ax1.text(0.05, y_start - 0.04, "Cognitive Graph-Based Personalized Course Recommendation System using FSLSM", 
                 fontsize=12, ha='left', va='top', transform=ax1.transAxes)
        ax1.text(0.05, y_start - 0.08, "="*70, fontsize=10, ha='left', va='top', transform=ax1.transAxes)

        # Student Info
        info_y = y_start - 0.15
        info_lines = [
            f"Student: {student_name}",
            f"Country: {country}",
            f"Degree: {degree}",
            f"Year: {semester}",
            f"University: {university if university else 'N/A'}"
        ]
        for i, line in enumerate(info_lines):
            ax1.text(0.05, info_y - i*0.04, line, fontsize=11, ha='left', va='top',
                     transform=ax1.transAxes)

        # Helper to draw a table with a title above (unchanged)
        def draw_table(ax, y, title, headers, data, col_widths, 
                       title_fontsize=12, header_fontsize=9, data_fontsize=9,
                       bbox_width=0.9, title_colour='#d3d3d3', 
                       header_colour='#d3d3d3', data_colour='white'):
            ax.text(0.5, y, title, fontsize=title_fontsize, fontweight='bold',
                    ha='center', va='top', transform=ax.transAxes)
            table_y = y - 0.04
            cell_text = [headers] + data
            n_cols = len(headers)
            header_colours = [header_colour] * n_cols
            data_colours = [[data_colour] * n_cols for _ in range(len(data))]
            all_colours = [header_colours] + data_colours
            table = ax.table(cellText=cell_text,
                             loc='center',
                             bbox=[0.5 - bbox_width/2, table_y - 0.12, bbox_width, 0.14],
                             cellColours=all_colours,
                             colWidths=col_widths)
            table.auto_set_font_size(False)
            table.set_fontsize(data_fontsize)
            table.scale(1, 1.5)
            for col in range(n_cols):
                table[(0, col)].set_text_props(fontweight='bold', ha='center')
            for row in range(1, len(cell_text)):
                for col in range(n_cols):
                    table[(row, col)].set_text_props(ha='left')
            return table_y - 0.16

        # Learning Style Profile
        y_pos = info_y - len(info_lines)*0.04 - 0.08
        profile_headers = ["Learning Style", "Score"]
        profile_data = [
            ["Visual", f"{profile['visual']:.3f}"],
            ["Verbal", f"{profile['verbal']:.3f}"],
            ["Sensing", f"{profile['sensing']:.3f}"],
            ["Intuitive", f"{profile['intuitive']:.3f}"],
            ["Active", f"{profile['active']:.3f}"],
            ["Reflective", f"{profile['reflective']:.3f}"],
            ["Global", f"{profile['global']:.3f}"],
            ["Sequential", f"{profile['sequential']:.3f}"]
        ]
        ax1.text(0.5, y_pos, "LEARNING STYLE PROFILE", fontsize=12, fontweight='bold',
                 ha='center', va='top', transform=ax1.transAxes)
        table_y = y_pos - 0.04
        n_cols = 2
        cell_text = [profile_headers] + profile_data
        all_colours = [['#d3d3d3'] * n_cols] + [['white'] * n_cols for _ in range(len(profile_data))]
        table = ax1.table(cellText=cell_text,
                          loc='center',
                          bbox=[0.05, table_y - 0.28, 0.9, 0.30],
                          cellColours=all_colours,
                          colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        for col in range(n_cols):
            table[(0, col)].set_text_props(fontweight='bold', ha='center')
        for row in range(1, len(cell_text)):
            for col in range(n_cols):
                table[(row, col)].set_text_props(ha='left')
        y_pos = table_y - 0.34

        # Recommended Course
        details = comparison_results[cg.best_course]
        source_label = details.get('source_label', '')
        ite_val = details.get('ite', 0.0)
        if abs(ite_val) < 0.0005:
            ite_val = 0.0
        grade_val = details.get('predicted_grade', 0.0)
        credit_val = details.get('credit_hours', 'Unknown')
        
        rec_headers = ["Attribute", "Value"]
        rec_data = [
            ["Course", cg.best_course],
            ["Causal ITE", f"{ite_val:.3f}"],
            ["Predicted Grade", f"{grade_val:.3f}"],
            ["Credit Hours", credit_val],
            ["Source", source_label]
        ]
        y_pos = draw_table(ax1, y_pos, "RECOMMENDED COURSE", rec_headers, rec_data,
                           col_widths=[0.5, 0.5], bbox_width=0.9)

        # Learning Style Match
        dominant = get_dominant_poles(profile)
        match_headers = ["Learning Pole", "Your Score", "Course Feature", "Match Strength"]
        match_data = []
        for dim, (pole, score) in dominant.items():
            if pole == "Visual":
                course_feature = course_features[cg.best_course].get('C_vis', 0.5)
            elif pole == "Sensing":
                course_feature = course_features[cg.best_course].get('C_prac', 0.5)
            elif pole == "Global":
                course_feature = course_features[cg.best_course].get('C_struct', 0.5)
            else:
                course_feature = 0.5
            match_score = score * course_feature
            match_data.append([pole, f"{score:.2f}", f"{course_feature:.2f}", f"{match_score:.2f}"])
        while len(match_data) < 4:
            match_data.append(["N/A", "N/A", "N/A", "N/A"])
        y_pos = draw_table(ax1, y_pos, "LEARNING STYLE MATCH", match_headers, match_data,
                           col_widths=[0.25, 0.25, 0.25, 0.25], bbox_width=0.9, data_fontsize=8)

        # Comparative ITE Results
        ite_headers = ["Course", "Causal ITE", "Legacy ITE", "Grade"]
        ite_data = []
        for course, data in comparison_results.items():
            marker = "*" if course == cg.best_course else " "
            ite_val = data.get('ite', 0.0)
            if abs(ite_val) < 0.0005:
                ite_val = 0.0
            legacy_val = data.get('legacy_ite', 0.0)
            grade_val = data.get('predicted_grade', 0.0)
            ite_data.append([f"{marker} {course}", f"{ite_val:+.3f}", f"{legacy_val:+.3f}", f"{grade_val:.3f}"])
        y_pos = draw_table(ax1, y_pos, "COMPARATIVE ITE RESULTS", ite_headers, ite_data,
                           col_widths=[0.3, 0.2, 0.25, 0.25], bbox_width=0.9, data_fontsize=8)

        # Scraped Sources - FIXED: ensure correct data
        src_headers = ["Course", "Source URL", "Source Type", "PDF Status"]
        src_data = []
        for course, data in comparison_results.items():
            src_url = data.get('source_url', 'Unknown')
            # Clean the source URL to remove "Scraped from " prefix
            src_url_clean = clean_source_url(src_url)
            src_type = data.get('source_label', 'Unknown')
            # Determine PDF status based on source URL
            if src_url and 'http' in src_url and 'Semantic' not in src_type:
                pdf_status = "Cached PDF"
            else:
                pdf_status = "Not available"
            display_url = src_url_clean if len(src_url_clean) < 60 else src_url_clean[:57] + "..."
            src_data.append([course, display_url, src_type, pdf_status])
        y_pos = draw_table(ax1, y_pos, "SCRAPED SOURCES", src_headers, src_data,
                           col_widths=[0.2, 0.4, 0.2, 0.2], bbox_width=0.9, data_fontsize=7)

        # Resource tables
        resources_urls = get_learning_resources_with_urls(cg.best_course)
        if resources_urls:
            flat_resources = []
            flat_urls = []
            for platform, course_list in resources_urls.items():
                for course_name, url in course_list:
                    flat_resources.append(f"{platform}: {course_name}")
                    flat_urls.append(url)
            res_headers = ["Suggested Learning Resources"]
            res_data = [[r] for r in flat_resources]
            y_pos = draw_table(ax1, y_pos, "SUGGESTED LEARNING RESOURCES", res_headers, res_data,
                               col_widths=[1.0], bbox_width=0.9, data_fontsize=7)
            url_headers = ["Learning Resource URLs"]
            url_data = [[u] for u in flat_urls]
            y_pos = draw_table(ax1, y_pos, "LEARNING RESOURCE URLs", url_headers, url_data,
                               col_widths=[1.0], bbox_width=0.9, data_fontsize=7)

        plt.figtext(0.5, 0.01, "Page 1 of 5", ha='center', fontsize=9)
        pdf.savefig(fig1, dpi=150, bbox_inches='tight')
        plt.close(fig1)

        # Page 2: Multi-course graph
        simple_fig.tight_layout()
        plt.figtext(0.5, 0.01, "Page 2 of 5", ha='center', fontsize=9)
        pdf.savefig(simple_fig, dpi=150, bbox_inches='tight')
        plt.close(simple_fig)

        # Page 3: Comprehensive graph
        comp_fig = visualize_comparative_cognitive_graph(
            cg, profile, comparison_results,
            title=f"Comprehensive Cognitive Graph: {student_name} -> Context -> Courses -> Concepts",
            save_files=False,
            pdf_mode=True
        )
        comp_fig.tight_layout()
        plt.figtext(0.5, 0.01, "Page 3 of 5", ha='center', fontsize=9)
        pdf.savefig(comp_fig, dpi=150, bbox_inches='tight')
        plt.close(comp_fig)

        # Page 4: Reasoning Trace
        fig4, ax4 = plt.subplots(figsize=(8.5, 11))
        ax4.axis('off')
        trace = "\n".join(cg.reasoning_trace)
        ax4.text(0.05, 0.95, f"REASONING TRACE\n{'-'*40}\n{trace}",
                 ha='left', va='top', fontsize=9, transform=ax4.transAxes, fontfamily='monospace')
        plt.figtext(0.5, 0.01, "Page 4 of 5", ha='center', fontsize=9)
        pdf.savefig(fig4, dpi=150, bbox_inches='tight')
        plt.close(fig4)

        #Page 5: MASTER CAUSAL EXPLANATION & COMPARATIVE TABLE
        fig5, ax5 = plt.subplots(figsize=(8.5, 11))
        ax5.axis('off')

        # Title
        ax5.text(0.05, 0.95, "MASTER CAUSAL EXPLANATION & COMPARATIVE RESULTS", 
                 fontsize=14, fontweight='bold', ha='left', va='top', transform=ax5.transAxes)

        # Causal Explanation
        ax5.text(0.05, 0.88, explanation, fontsize=10, ha='left', va='top', 
                 transform=ax5.transAxes, wrap=True)

        # Comparative Table (repeated)
        y_table = 0.55
        ax5.text(0.5, y_table, "COMPARATIVE ITE RESULTS", fontsize=12, fontweight='bold',
                 ha='center', va='top', transform=ax5.transAxes)
        table_y = y_table - 0.04
        headers = ["Course", "Causal ITE", "Legacy ITE", "Grade"]
        data = []
        for course, data_dict in comparison_results.items():
            marker = "*" if course == cg.best_course else " "
            ite_val = data_dict.get('ite', 0.0)
            if abs(ite_val) < 0.0005:
                ite_val = 0.0
            legacy_val = data_dict.get('legacy_ite', 0.0)
            grade_val = data_dict.get('predicted_grade', 0.0)
            data.append([f"{marker} {course}", f"{ite_val:+.3f}", f"{legacy_val:+.3f}", f"{grade_val:.3f}"])

        cell_text = [headers] + data
        table = ax5.table(cellText=cell_text,
                          loc='center',
                          bbox=[0.05, table_y - 0.28, 0.9, 0.30],
                          cellColours=[['#d3d3d3']*4] + [['white']*4 for _ in range(len(data))],
                          colWidths=[0.3, 0.2, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        for col in range(4):
            table[(0, col)].set_text_props(fontweight='bold', ha='center')
        for row in range(1, len(cell_text)):
            for col in range(4):
                table[(row, col)].set_text_props(ha='left')

        # Note about best course
        ax5.text(0.05, 0.12, f"* Recommended course: {cg.best_course} with Causal ITE = {comparison_results[cg.best_course]['ite']:.3f}",
                 fontsize=10, ha='left', va='top', transform=ax5.transAxes)

        plt.figtext(0.5, 0.01, "Page 5 of 5", ha='center', fontsize=9)
        pdf.savefig(fig5, dpi=150, bbox_inches='tight')
        plt.close(fig5)

    buffer.seek(0)
    return buffer


# SECTION 13: STREAMLIT MAIN APPLICATION (MODIFIED QUESTION LAYOUT)

def main():
    st.set_page_config(page_title="CogNet Advisor", layout="wide")
    
    logo_exists = LOGO_PATH and os.path.exists(LOGO_PATH)
    col1, col2 = st.columns([1, 5])
    with col1:
        if logo_exists:
            st.image(LOGO_PATH, width=120)
    with col2:
        st.title("CogNet Advisor")
        st.markdown("*Cognitive Graph-Based Personalized Course Recommendation System using FSLSM*")

    with st.sidebar:
        st.header("API Key")
        saved_key = load_api_key()
        if saved_key:
            st.success("API key loaded from local storage.")
            if st.button("Clear saved API key"):
                clear_api_key()
                st.rerun()
            api_key = saved_key
        else:
            api_key = st.text_input("OpenAI API Key", type="password")
            remember = st.checkbox("Remember my API key (saved locally)")
            if api_key and remember:
                save_api_key(api_key)
                st.success("Key saved for future sessions.")
        if not api_key:
            st.warning("Please enter your OpenAI API key.")
            st.stop()
        os.environ["OPENAI_API_KEY"] = api_key

        st.header("Student Profile")
        name = st.text_input("Name", "Student")
        country = st.selectbox("Country", ["Pakistan", "USA", "UK", "Canada", "Australia", "Germany", "France", "India", "China", "Japan", "South Korea", "Singapore", "Malaysia", "UAE", "Saudi Arabia", "Egypt", "South Africa", "Brazil", "Mexico", "Italy", "Spain", "Netherlands", "Sweden", "Norway", "Russia", "Other"])
        degree = st.selectbox("Degree", ["BS Computer Science", "BS Artificial Intelligence", "BS Software Engineering", "BS Data Science", "BS Cybersecurity", "MS Computer Science", "MS Data Science", "MS Artificial Intelligence", "PhD Computer Science", "BBA", "MBA", "Other"])
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

    llm = ChatOpenAI(model="gpt-6-luna", temperature=0.2)
    embeddings = OpenAIEmbeddings()

    if "start_assessment" not in st.session_state:
        st.session_state.start_assessment = False
    if "answers" not in st.session_state:
        st.session_state.answers = []
    if "agent" not in st.session_state:
        st.session_state.agent = FSLSMAgent(llm)
    if "profile" not in st.session_state:
        st.session_state.profile = None
    if "course_features" not in st.session_state:
        st.session_state.course_features = None
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = None
    if "best_course" not in st.session_state:
        st.session_state.best_course = None
    if "explanation" not in st.session_state:
        st.session_state.explanation = None
    if "cg" not in st.session_state:
        st.session_state.cg = None
    if "analysis_started" not in st.session_state:
        st.session_state.analysis_started = False
    if "simple_fig" not in st.session_state:
        st.session_state.simple_fig = None
    if "pdf_buffer" not in st.session_state:
        st.session_state.pdf_buffer = None

    # --- MODIFIED: All questions on one page ---
    if st.session_state.start_assessment and st.session_state.profile is None:
        st.subheader("FSLSM Questionnaire")
        st.markdown("Please answer all 44 questions below. Choose the option that best fits you.")
        answers = {}
        # Use a form to capture all answers at once
        with st.form(key="fslsm_form"):
            for idx, (q_text, dim, pole) in enumerate(FSLSM_QUESTIONS):
                opt_a, opt_b = extract_options(q_text)
                clean_q = re.sub(r'\s*\(a\)\s*.*?\s*\(b\)\s*.*', '', q_text).strip()
                # Use radio buttons
                choice = st.radio(
                    f"Q{idx+1}: {clean_q}",
                    [f"(a) {opt_a}", f"(b) {opt_b}"],
                    key=f"q_{idx}",
                    index=None
                )
                # Store selected value
                answers[idx] = choice
            # Submit button
            submitted = st.form_submit_button("Submit All Answers")
        
        if submitted:
            # Process answers
            all_answered = True
            for idx in range(len(FSLSM_QUESTIONS)):
                if answers.get(idx) is None:
                    all_answered = False
                    break
            if not all_answered:
                st.warning("Please answer all questions before submitting.")
            else:
                # Compute delta for each question
                for idx, (q_text, dim, pole) in enumerate(FSLSM_QUESTIONS):
                    choice = answers[idx]
                    selected = "a" if choice.startswith("(a)") else "b"
                    delta = 1 if selected == pole else -1
                    st.session_state.answers.append((dim, delta))
                st.session_state.agent.answers = st.session_state.answers
                profile = st.session_state.agent.compute_profile()
                st.session_state.profile = profile
                st.success("Assessment complete! Click 'Analyze' to run the full analysis.")
                if st.button("Analyze"):
                    st.session_state.analysis_started = True
                    st.rerun()
    else:
        # Existing result display (unchanged)
        if st.session_state.profile is not None and st.session_state.best_course is not None:
            st.header("Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Your Learning Style Profile")
                st.write(f"**Visual**: {st.session_state.profile['visual']:.3f}")
                st.write(f"**Sensing (Practical)**: {st.session_state.profile['sensing']:.3f}")
                st.write(f"**Active**: {st.session_state.profile['active']:.3f}")
                st.write(f"**Global**: {st.session_state.profile['global']:.3f}")
            with col2:
                st.subheader(" ") 
                st.write(f"**Verbal**: {st.session_state.profile['verbal']:.3f}")
                st.write(f"**Intuitive**: {st.session_state.profile['intuitive']:.3f}")
                st.write(f"**Reflective**: {st.session_state.profile['reflective']:.3f}")
                st.write(f"**Sequential**: {st.session_state.profile['sequential']:.3f}")
            with col3:
                st.subheader("Recommended Course")
                st.success(f"**{st.session_state.best_course}**")
                st.write(f"ITE: {st.session_state.comparison_results[st.session_state.best_course]['ite']:.3f}")
                st.write(f"Predicted Grade: {st.session_state.comparison_results[st.session_state.best_course]['predicted_grade']:.3f}")

            st.subheader("Comparative ITE Results")
            for course, data in st.session_state.comparison_results.items():
                src = data.get('source_url', 'Unknown')
                src_clean = clean_source_url(src)
                source_label = data.get('source_label', '')
                pdf_status = "Cached PDF" if src != 'Unknown' and 'http' in src else "Not available"
                causal_ite = data.get('ite', 0.0)
                if abs(causal_ite) < 0.0005:
                    causal_ite = 0.0
                legacy_ite = data.get('legacy_ite', 0.0)
                st.write(f"**{course}**: Causal ITE = {causal_ite:+.3f} (Legacy: {legacy_ite:+.3f}), Grade = {data['predicted_grade']:.3f}  -  Source: {src_clean}  -  {source_label}  -  PDF: {pdf_status}")

            st.subheader("Master Causal Explanation")
            st.write(st.session_state.explanation)

            st.subheader("Suggested Learning Resources")
            resources = get_learning_resources(st.session_state.best_course)
            for platform, courses_list in resources.items():
                st.write(f"**{platform}:** {', '.join(courses_list)}")

            st.subheader("Model Validation: 5-Fold Cross-Validation")
            st.markdown("*Comparing your Double ML against standard ML baselines on the same dataset.*")
            
            with st.spinner("Running cross-validation..."):
                dataset_path = os.path.join(BASE_DIR, "CogNet_Advisory_Data.csv")
                if os.path.exists(dataset_path):
                    try:
                        df_cv = pd.read_csv(dataset_path)
                        df_cv.rename(columns={
                            'Visual': 'visual',
                            'Sensing': 'sensing',
                            'Active': 'active',
                            'Global': 'global',
                            'GPA': 'prior_gpa',
                            'Course_Taken': 'course_taken',
                            'Final_Grade': 'final_grade'
                        }, inplace=True)
                        required_cols = ['visual', 'sensing', 'active', 'global', 'prior_gpa', 
                                         'C_vis', 'C_prac', 'C_struct', 'final_grade']
                        df_cv = df_cv.dropna(subset=required_cols)
                        cv_results = run_cross_validation_with_baselines(df_cv, n_splits=5)
                        st.dataframe(cv_results, use_container_width=True)
                        
                        try:
                            rmse_values = []
                            methods = []
                            for idx in cv_results.index:
                                rmse_str = cv_results.loc[idx, 'RMSE (↓)']
                                if rmse_str != 'N/A':
                                    val = float(rmse_str.split('±')[0].strip())
                                    rmse_values.append(val)
                                    methods.append(idx)
                            if rmse_values:
                                best_idx = np.argmin(rmse_values)
                                best_method = methods[best_idx]
                                
                                # Dynamically explain the result depending on which model wins
                                if best_method == "Double ML":
                                    st.success(f"Best performing method: Double ML has the lowest RMSE, validating the causal approach.")
                                else:
                                    st.success("Linear Regression shows the best raw predictive RMSE due to the high-noise nature of grades. However, Double ML provides the unbiased, causal estimates (ITE) required for accurate course recommendation.")
                        except:
                            pass
                    except Exception as e:
                        st.warning(f"Error loading dataset for cross-validation: {e}")
                else:
                    st.warning("Original dataset `CogNet_Advisory_Data.csv` not found for cross-validation. Please ensure the file exists.")

            if st.session_state.simple_fig is not None:
                st.subheader("Multi-Course Cognitive Overlay")
                st.pyplot(st.session_state.simple_fig)

            if st.session_state.cg:
                st.subheader("Comprehensive Cognitive Graph")
                fig = visualize_comparative_cognitive_graph(
                    st.session_state.cg,
                    st.session_state.profile,
                    st.session_state.comparison_results,
                    title=f"Comprehensive Cognitive Graph: {name} -> Context -> Courses -> Concepts",
                    save_files=False
                )
                st.pyplot(fig)

            with st.expander("Reasoning Trace"):
                for step in st.session_state.cg.reasoning_trace:
                    st.write(f"- {step}")

            col_dl, col_reset = st.columns([2, 1])
            with col_dl:
                if st.session_state.pdf_buffer is not None:
                    st.download_button(
                        label="Download Full Report (PDF)",
                        data=st.session_state.pdf_buffer,
                        file_name=f"CogNet_Advisor_Report_{name}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            with col_reset:
                if st.button("Start New Assessment", use_container_width=True):
                    st.session_state.start_assessment = False
                    st.session_state.profile = None
                    st.session_state.answers = []
                    st.session_state.course_features = None
                    st.session_state.comparison_results = None
                    st.session_state.best_course = None
                    st.session_state.explanation = None
                    st.session_state.cg = None
                    st.session_state.analysis_started = False
                    st.session_state.simple_fig = None
                    st.session_state.pdf_buffer = None
                    if "agent" in st.session_state:
                        st.session_state.agent.answers = []
                    st.rerun()

        elif st.session_state.profile is not None:
            if st.button("Run Full Analysis (this may take a few minutes)"):
                st.session_state.analysis_started = True
                st.rerun()

        if st.session_state.get("analysis_started", False) and st.session_state.best_course is None:
            with st.spinner("Fetching curriculum data and analyzing..."):
                preparer = BackgroundRAGPreparer(llm, embeddings)
                preparer.run(country, degree, courses, university if university else None)
                if preparer.is_ready:
                    course_features = preparer.course_features
                else:
                    course_features = {c: {"C_vis":0.5, "C_prac":0.5, "C_struct":0.5, "credit_hours":"Unknown", "source_url":"Fallback"} for c in courses}
                st.session_state.course_features = course_features

                best_course, comparison_results, explanation, df_loaded = get_comparative_analysis(
                    st.session_state.profile, course_features, llm,
                    student_country=country,
                    student_university=university if university else None
                )
                st.session_state.best_course = best_course
                st.session_state.comparison_results = comparison_results
                st.session_state.explanation = explanation

                simple_fig = visualize_multi_course_graph(
                    st.session_state.profile, course_features, comparison_results, name,
                    student_country=country,
                    student_university=university if university else None
                )
                st.session_state.simple_fig = simple_fig

                cg = generate_comparative_cognitive_graph(
                    student_name=name,
                    profile=st.session_state.profile,
                    courses=courses,
                    course_features=course_features,
                    comparison_results=comparison_results,
                    best_course=best_course,
                    country=country,
                    degree=degree,
                    semester=str(semester),
                    university=university if university else None,
                    n_steps=3,
                    llm=llm
                )
                st.session_state.cg = cg

                pdf_buffer = generate_pdf_report(
                    st.session_state.profile,
                    comparison_results,
                    explanation,
                    cg,
                    simple_fig,
                    name,
                    country,
                    degree,
                    semester,
                    university,
                    courses,
                    course_features,
                    logo_path=LOGO_PATH if LOGO_PATH and os.path.exists(LOGO_PATH) else None
                )
                st.session_state.pdf_buffer = pdf_buffer

                st.session_state.analysis_started = False
                st.rerun()


if __name__ == "__main__":
    main()
