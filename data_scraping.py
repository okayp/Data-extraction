import time
import re
import os
from urllib.parse import urlparse
from datetime import datetime
import pymongo
import traceback
import streamlit as st
from fastapi import FastAPI, HTTPException
import time
import re
import os
from urllib.parse import urlparse
from datetime import datetime
import pymongo
import traceback
import json
import pycountry
import phonenumbers
import spacy
from bs4 import BeautifulSoup
from typing import Union, List, Optional, Dict, Any
from pydantic import BaseModel, Field

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from bs4 import BeautifulSoup
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from pydantic import BaseModel, Field
from langchain_ollama import OllamaLLM
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
import time
import re
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from urllib.parse import urlparse
from datetime import datetime
import pymongo
import traceback
import json
import pycountry
import phonenumbers
import spacy
from bs4 import BeautifulSoup
from typing import Union, List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_ollama import OllamaLLM
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
import sqlite3
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import random
import json


def scrape_linkedin_company(company_name, display_name=None, email=None, password=None, save_to_file=False):
    """
    Scrape a single company's data from LinkedIn using Selenium.

    Args:
        company_name (str): The company name as it appears in the LinkedIn URL (e.g., "google")
        display_name (str, optional): The display name of the company (e.g., "Google"). Defaults to company_name.
        email (str, optional): LinkedIn login email. Required for login.
        password (str, optional): LinkedIn login password. Required for login.
        save_to_file (bool, optional): Whether to save data to JSON file. Defaults to False.

    Returns:
        dict: The scraped company data
    """
    # Use company_name as display_name if not provided
    if display_name is None:
        display_name = company_name.capitalize()

    # Check for required credentials
    if email is None or password is None:
        raise ValueError("LinkedIn email and password are required")

    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_argument("--disable-notifications")

    # Initialize driver and company_data
    driver = webdriver.Chrome(options=chrome_options)
    company_data = {}

    try:
        # LINKEDIN LOGIN
        print("Logging into LinkedIn...")
        driver.get("https://www.linkedin.com/login")
        time.sleep(3)

        # Enter email
        email_field = driver.find_element(By.ID, "username")
        email_field.clear()
        for char in email:  # Type like a human
            email_field.send_keys(char)
            time.sleep(random.uniform(0.05, 0.15))

        # Enter password
        password_field = driver.find_element(By.ID, "password")
        password_field.clear()
        for char in password:  # Type like a human
            password_field.send_keys(char)
            time.sleep(random.uniform(0.05, 0.15))

        # Click sign in
        sign_in_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
        sign_in_button.click()

        # Wait for login to complete
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "global-nav"))
            )
            print("Login successful")
            # Wait a bit longer after login to ensure session is established
            time.sleep(random.uniform(3, 5))
        except TimeoutException:
            print("Login timed out - check if login was successful")

        # EXTRACT COMPANY DATA
        # Go directly to the About page
        about_url = f"https://www.linkedin.com/company/{company_name}/about/"
        print(f"\nVisiting {display_name} at {about_url}")
        driver.get(about_url)
        time.sleep(random.uniform(5, 8))  # Random wait to avoid detection

        # Check if redirected to login
        if "login" in driver.current_url or "authenticate" in driver.current_url:
            print("Redirected to login page, attempting login again...")
            driver.get("https://www.linkedin.com/login")
            time.sleep(3)

            # Re-enter credentials
            email_field = driver.find_element(By.ID, "username")
            email_field.clear()
            for char in email:
                email_field.send_keys(char)
                time.sleep(random.uniform(0.05, 0.15))

            password_field = driver.find_element(By.ID, "password")
            password_field.clear()
            for char in password:
                password_field.send_keys(char)
                time.sleep(random.uniform(0.05, 0.15))

            sign_in_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
            sign_in_button.click()

            # Wait and navigate back to company page
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "global-nav"))
            )
            driver.get(about_url)
            time.sleep(random.uniform(5, 8))

        # Extract essential company data using JavaScript
        company_data = driver.execute_script("""
            function extractCompanyBasics() {
                // Initialize data object with essential fields only
                const data = {
                    "Company Name": null,
                    "Description": null,
                    "Industry": null,
                    "Company size": null,
                    "Headquarters": null,
                    "Website": null,
                    "Specialties": [],
                    "Founded": null,
                    "URL": window.location.href
                };

                // Extract company name (usually in h1)
                const nameElements = document.querySelectorAll('h1');
                if (nameElements.length > 0) {
                    data["Company Name"] = nameElements[0].textContent.trim();
                }

                // Helper function to find field values by labels
                function findFieldByLabel(label) {
                    // Method 1: Look for dt/dd pairs
                    const dtElements = Array.from(document.querySelectorAll('dt'));
                    for (const dt of dtElements) {
                        if (dt.textContent.trim().toLowerCase() === label.toLowerCase()) {
                            const dd = dt.nextElementSibling;
                            if (dd && dd.tagName === 'DD') {
                                return dd.textContent.trim();
                            }
                        }
                    }

                    // Method 2: Look for labeled sections
                    const labelElements = Array.from(document.querySelectorAll('span, div, h3'));
                    for (const el of labelElements) {
                        if (el.textContent.trim().toLowerCase() === label.toLowerCase()) {
                            let parent = el.parentElement;
                            let next = parent.nextElementSibling;
                            if (next) {
                                return next.textContent.trim();
                            }
                        }
                    }

                    // Method 3: Look for text patterns like "Label: Value"
                    const allElements = document.querySelectorAll('p, div, span');
                    for (const el of allElements) {
                        const text = el.textContent.trim();
                        const regex = new RegExp(`${label}[:\\s]+([^\\n]+)`, 'i');
                        const match = text.match(regex);
                        if (match && match[1]) {
                            return match[1].trim();
                        }
                    }

                    return null;
                }

                // Extract each field
                const fields = [
                    ["Industry", "Industry"],
                    ["Company size", "Company size"],
                    ["Headquarters", "Headquarters"],
                    ["Founded", "Founded"],
                    ["Website", "Website"]
                ];

                for (const [dataKey, label] of fields) {
                    const value = findFieldByLabel(label);
                    if (value) {
                        data[dataKey] = value;
                    }
                }

                // Extract description
                // Method 1: Standard description classes
                const descElements = document.querySelectorAll('.org-about-us__description, .break-words p');
                if (descElements.length > 0) {
                    data["Description"] = descElements[0].textContent.trim();
                } else {
                    // Method 2: Look for paragraphs in the about page
                    const paragraphs = document.querySelectorAll('p');
                    for (const p of paragraphs) {
                        const text = p.textContent.trim();
                        if (text.length > 50) { // Likely a description if it's a longer paragraph
                            data["Description"] = text;
                            break;
                        }
                    }
                }

                // Special handling for specialties (comma-separated)
                const specialtiesText = findFieldByLabel("Specialties");
                if (specialtiesText) {
                    data["Specialties"] = specialtiesText.split(',')
                        .map(s => s.trim())
                        .filter(s => s.length > 0);
                }

                return data;
            }

            return extractCompanyBasics();
        """)

        # Print extracted data
        print(f"\n--- Extracted Data for {display_name} ---")
        for key, value in company_data.items():
            if key != "Specialties":
                if value is None or value == "":
                    value = "Not found"
                print(f"{key}: {value}")
            else:
                if not value:
                    print(f"{key}: []")
                else:
                    print(f"{key}: {', '.join(value)}")

        # Save data to JSON file if requested
        if save_to_file:
            output_file = f"linkedin_{company_name}_data.json"
            with open(output_file, "w") as f:
                json.dump(company_data, f, indent=2)
            print(f"\nCompany data saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")
        company_data = {"error": str(e)}

    finally:
        # Close the browser
        driver.quit()

    return company_data
def init_db():
    conn = sqlite3.connect("results.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            llm_output TEXT,
            bert_output TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def insert_result(url, llm_output, bert_output):
    conn = sqlite3.connect("results.db")
    c = conn.cursor()
    c.execute('''
        INSERT INTO results (url, llm_output, bert_output)
        VALUES (?, ?, ?)
    ''', (url, llm_output, bert_output))
    conn.commit()
    conn.close()

def get_all_results():
    conn = sqlite3.connect("results.db")
    c = conn.cursor()
    c.execute('SELECT url, llm_output, bert_output, timestamp FROM results ORDER BY timestamp DESC')
    rows = c.fetchall()
    conn.close()
    return rows

class CompanyInfo2(BaseModel):
    name: str
    industry: str
    headquarters: str
    company_size: str
    website: str
    country: str
    phone_number: str
    revenue: str
    # Additional fields that might be extracted
    email: Optional[str] = None
    social_media: Optional[Dict[str, str]] = None
    cities: Optional[List[str]] = None
    states: Optional[List[str]] = None
    founding_year: Optional[str] = None
    specialties: Optional[List[str]] = None
    description: Optional[str] = None
    address: Optional[str] = None
    company_type: Optional[str] = None
    # Track which model was used for extraction
    extraction_model: Optional[str] = None
from keybert import KeyBERT
class CompanyInfo(BaseModel):
    name: Optional[Union[str, List[str]]] = Field(default=None,description="Company name")
    industry: Optional[Union[str, List[str]]] = Field(default=None,description="Company industry")
    headquarters: Optional[Union[str, List[str]]] = Field(default=None,description="Headquarters location")
    company_size: Optional[Union[str, List[str]]] = Field(default=None,description="Number of employees")
    website:Optional[Union[str,List[str]]]=Field(default=None,description="Give the website of the company")
    country:Optional[Union[str,List[str]]]=Field(default=None,description="Give the country of the company")
    email_address:Optional[Union[str,List[str]]]=Field(default=None,description="Give the email address of the company if available")
    social_media:Optional[Union[str,List[str]]]=Field(default=None,description="Give any social media sites if mentioned")
    phone_number:Optional[Union[str,List[str]]]=Field(default=None,description="Give the phone number of company if  available")
    founding_year:Optional[Union[str,List[str]]]=Field(default=None,description="Give the founding year of company if available")
    revenue:Optional[Union[str,List[str]]]=Field(default=None,description="Give the revenue of the company if mentioned")
    extraction_model: Optional[str] = None



# Normalize URL
def normalize_url(url):
    if not url:
        return None

    url = str(url).strip()
    if not url:
        return None

    if url.lower() in ['data', 'null', 'n/a', 'none', '']:
        return None

    clean_url = url.lower()
    if clean_url.startswith('http://'):
        clean_url = clean_url[7:]
    elif clean_url.startswith('https://'):
        clean_url = clean_url[8:]

    if clean_url.startswith('www.'):
        clean_url = clean_url[4:]

    normalized_url = f"https://www.{clean_url}"

    return normalized_url


# Scrape website
def scrape_website(url, driver, timeout=20):
    results = []
    results.append(f"Website: {url}")
    results.append(f"Scrape Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    driver.set_page_load_timeout(timeout)

    try:
        driver.get(url)
        time.sleep(5)

        try:
            cookie_buttons = [
                "CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll",
                "accept-cookies",
                "cookie_bar_allow_all"
            ]

            for button_id in cookie_buttons:
                try:
                    cookie_btn = WebDriverWait(driver, 3).until(
                        EC.element_to_be_clickable((By.ID, button_id))
                    )
                    cookie_btn.click()
                    time.sleep(1)
                    break
                except:
                    continue

            button_texts = ["Accept", "Accept All", "Akzeptieren", "Accepteren"]
            for text in button_texts:
                try:
                    xpath = f"//button[contains(text(), '{text}')]"
                    button = WebDriverWait(driver, 2).until(
                        EC.element_to_be_clickable((By.XPATH, xpath))
                    )
                    button.click()
                    break
                except:
                    continue
        except Exception:
            pass

        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        text = soup.get_text()

        results.append("\n--- HOME PAGE INFORMATION ---")

        title_element = soup.find('title')
        if title_element:
            company_name = re.sub(r'\s*[-|]\s*.*$', '', title_element.text).strip()
            results.append(f"Company Name: {company_name}")

        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            results.append(f"Meta Description: {meta_desc['content']}")

        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_matches = re.findall(email_pattern, text)

        phone_pattern = r'(?:\+\d{1,4}[-.\s]?)?(?:\(?\d{1,4}\)?[-.\s]?)?(?:\d{1,4}[-.\s]?){1,3}\d{1,4}'
        phone_matches = re.findall(phone_pattern, text)

        about_headings = ['about', 'about us', 'who we are', 'our story']
        about_content = []

        for heading in about_headings:
            elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'],
                                     string=lambda text: text and heading.lower() in text.lower())

            for element in elements:
                next_element = element.find_next('p')
                if next_element and len(next_element.text.strip()) > 20:
                    about_content.append(next_element.text.strip())

        about_phrases = ['founded in', 'established in', 'our mission', 'we believe']
        for phrase in about_phrases:
            matches = re.findall(f'[^.!?]*{phrase}[^.!?]*[.!?]', text, re.IGNORECASE)
            about_content.extend(matches)

        about_page_visited = False
        about_keywords = ['about', 'about-us', 'aboutus', 'who-we-are', 'company', 'over-ons', 'über-uns']

        for keyword in about_keywords:
            try:
                links = driver.find_elements(By.XPATH,
                                             f"//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{keyword.lower()}')]")

                if not links:
                    links = driver.find_elements(By.XPATH, f"//a[contains(@href, '{keyword.lower()}')]")

                if links:
                    links[0].click()
                    time.sleep(3)
                    about_page_visited = True

                    results.append("\n--- ABOUT PAGE INFORMATION ---")
                    about_soup = BeautifulSoup(driver.page_source, 'html.parser')
                    about_text = about_soup.get_text()

                    about_paragraphs = about_soup.find_all('p')
                    about_page_content = []

                    for p in about_paragraphs:
                        if len(p.text.strip()) > 50:
                            about_page_content.append(p.text.strip())

                    if about_page_content:
                        results.append("About Content:")
                        for content in about_page_content[:3]:
                            results.append(f"- {content}")

                    about_email_matches = re.findall(email_pattern, about_text)
                    email_matches.extend(about_email_matches)

                    about_phone_matches = re.findall(phone_pattern, about_text)
                    phone_matches.extend(about_phone_matches)

                    driver.back()
                    time.sleep(2)
                    break

            except Exception:
                pass

        if not about_page_visited and about_content:
            results.append("\n--- ABOUT INFORMATION (from Home Page) ---")
            results.append("About Content:")
            for content in about_content[:3]:
                results.append(f"- {content}")

        contact_page_visited = False
        contact_keywords = ['contact', 'contact-us', 'contactus', 'get-in-touch', 'kontakt']

        for keyword in contact_keywords:
            try:
                links = driver.find_elements(By.XPATH,
                                             f"//a[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{keyword.lower()}')]")

                if not links:
                    links = driver.find_elements(By.XPATH, f"//a[contains(@href, '{keyword.lower()}')]")

                if links:
                    links[0].click()
                    time.sleep(3)
                    contact_page_visited = True

                    results.append("\n--- CONTACT PAGE INFORMATION ---")
                    contact_soup = BeautifulSoup(driver.page_source, 'html.parser')
                    contact_text = contact_soup.get_text()

                    contact_email_matches = re.findall(email_pattern, contact_text)
                    email_matches.extend(contact_email_matches)

                    contact_phone_matches = re.findall(phone_pattern, contact_text)
                    phone_matches.extend(contact_phone_matches)

                    address_pattern = r'\b\d{4,5}[-\s]?[A-Za-z]*\s[A-Za-z]+'
                    address_matches = re.findall(address_pattern, contact_text)

                    social_patterns = {
                        'LinkedIn': r'linkedin\.com/(?:company|in)/([^/\s"\'>]+)',
                        'Facebook': r'facebook\.com/([^/\s"\'>]+)',
                        'Twitter': r'(?:twitter|x)\.com/([^/\s"\'>]+)',
                        'Instagram': r'instagram\.com/([^/\s"\'>]+)'
                    }

                    social_matches = {}
                    for platform, pattern in social_patterns.items():
                        matches = re.findall(pattern, contact_soup.prettify())
                        if matches:
                            social_matches[platform] = matches[0]

                    break

            except Exception:
                pass

        if not contact_page_visited:
            footer = soup.find('footer')
            if footer:
                footer_text = footer.get_text()
                footer_email_matches = re.findall(email_pattern, footer_text)
                email_matches.extend(footer_email_matches)

                footer_phone_matches = re.findall(phone_pattern, footer_text)
                phone_matches.extend(footer_phone_matches)

        if email_matches:
            unique_emails = list(dict.fromkeys(email_matches))
            results.append("\n--- CONTACT INFORMATION ---")
            results.append("Email Addresses:")
            for email in unique_emails:
                results.append(f"- {email}")

        if phone_matches:
            cleaned_phones = []
            for phone in phone_matches:
                cleaned = re.sub(r'[^\d+]', '', phone)
                if cleaned and len(cleaned) > 6:
                    cleaned_phones.append(cleaned)

            unique_phones = list(dict.fromkeys(cleaned_phones))
            if not "CONTACT INFORMATION" in results[-3:]:
                results.append("\n--- CONTACT INFORMATION ---")
            results.append("Phone Numbers:")
            for phone in unique_phones:
                results.append(f"- {phone}")

        if 'social_matches' in locals() and social_matches:
            if not "CONTACT INFORMATION" in results[-3:]:
                results.append("\n--- CONTACT INFORMATION ---")
            results.append("Social Media:")
            for platform, handle in social_matches.items():
                results.append(f"- {platform}: {handle}")

        if 'address_matches' in locals() and address_matches:
            if not "CONTACT INFORMATION" in results[-3:]:
                results.append("\n--- CONTACT INFORMATION ---")
            results.append("Potential Addresses:")
            unique_addresses = list(dict.fromkeys(address_matches))
            for address in unique_addresses:
                results.append(f"- {address}")

        return True, results

    except TimeoutException:
        results.append(f"\nError: Website loading timed out")
        return False, results

    except WebDriverException as e:
        results.append(f"\nError: WebDriver exception - {str(e)}")
        return False, results

    except Exception as e:
        results.append(f"\nError during scraping: {str(e)}")
        return False, results


def llm_model(scraped_text):
    query = scraped_text
    knowledge_base = [scraped_text]

    model = SentenceTransformer('all-MiniLM-L6-v2')
    kb_embeddings = model.encode(knowledge_base)

    index = faiss.IndexFlatL2(kb_embeddings.shape[1])
    index.add(np.array(kb_embeddings))

    query_embedding = model.encode([query])

    D, I = index.search(query_embedding, k=2)
    retrieved_docs = [knowledge_base[i] for i in I[0]]

    llm = OllamaLLM(model="deepseek-r1:7b")
    parser = PydanticOutputParser(pydantic_object=CompanyInfo)

    prompt = PromptTemplate(
        template="""
                Extract company information from the following text:
                Original Text: {original_text}
                Retrieved Data: {retrieved_data}

                - The output MUST be a valid JSON following this format:
                {{
                    "name": "Company Name",
                    "industry": "Industry of company",
                    "headquarters": "Headquarters Location",
                    "company_size": "Number of Employees",
                    "website": "Company Website URL",
                    "country": "Country in which company is located",
                    "email_address": "Give the email address of the company",
                    "social_media": "Give the social media of the company",
                    "phone_number": "Company Phone Number (if available, otherwise empty string)",
                    "founding_year": "Give the founding year of the company",
                    "revenue": "Revenue of company (if available, otherwise empty string)"
                }}

                - Ensure all fields are always present.
                - If a field is missing, use an empty string `""` instead of `null` or omitting it.

                {format_instructions}
            """,
        input_variables=["original_text", "retrieved_data"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | llm | parser

    result = chain.invoke({
        "original_text": query,
        "retrieved_data": " ".join(retrieved_docs)
    })

    return result

def bert_model(text, use_deepseek=False):
    def is_country(location):
        try:
            country_names = {country.name for country in pycountry.countries}
            country_abbreviations = {country.alpha_2 for country in pycountry.countries}
            return location in country_names or location in country_abbreviations
        except Exception as e:
            print(f"Error checking country: {str(e)}")
            return False

    def filter_entities(entities):
        try:
            org_entities = set()
            loc_entities = set()
            country_entities = set()
            city_entities = set()
            state_entities = set()

            for entity in entities:
                entity_text = entity['word'].strip()
                entity_group = entity['entity_group']
                entity_score = entity.get('score', 0.95)  # Default score if not present

                if '##' in entity_text:
                    continue

                if entity_group == 'ORG' and entity_score > 0.9:
                    org_entities.add(entity_text)
                elif entity_group in ['LOC', 'GPE', 'LOCATION']:
                    if is_country(entity_text):
                        country_entities.add(entity_text)
                    elif 'CITY' in entity_group or entity_text.lower().endswith('city'):
                        city_entities.add(entity_text)
                    elif 'STATE' in entity_group or entity_text in ['CA', 'NY', 'TX']:
                        state_entities.add(entity_text)
                    else:
                        loc_entities.add(entity_text)

            return {
                "organization": list(org_entities),
                "location": list(loc_entities),
                "country": list(country_entities),
                "city": list(city_entities),
                "state": list(state_entities)
            }
        except Exception as e:
            print(f"Error filtering entities: {str(e)}")
            return {
                "organization": [],
                "location": [],
                "country": [],
                "city": [],
                "state": []
            }

    def extract_employee_count_url_phone_pos(text):
        try:
            doc = nlp(text[:10000])
            employee_counts = set()
            urls = []
            phone_numbers = set()
            company_types = set()

            # Detect company type patterns
            company_type_patterns = {
                'LLC': r'\b(?:LLC|Limited Liability Company)\b',
                'Inc': r'\b(?:Inc|Incorporated)\b',
                'Corp': r'\b(?:Corp|Corporation)\b',
                'Ltd': r'\b(?:Ltd|Limited)\b',
                'GmbH': r'\bGmbH\b',
                'AG': r'\b(?:AG|Aktiengesellschaft)\b',
                'SA': r'\b(?:SA|S\.A\.)\b',
                'BV': r'\b(?:BV|B\.V\.)\b',
                'Privately Held': r'\bPrivately Held\b',
                'Public Company': r'\bPublic Company\b'
            }

            for company_type, pattern in company_type_patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    company_types.add(company_type)

            url_pattern = r"(https?://[^\s]+)"
            urls = re.findall(url_pattern, text)

            phone_numbers = set()

            # Regex pattern to match optional country code and up to 15 digits with separators
            phone_pattern = r"(?:\+?(\d{1,3})[ ]?)?([\d\s]{10,15})"

            raw_phone_numbers = re.findall(phone_pattern, text)

            for match in raw_phone_numbers:
                country_code, rest_number = match
                number_digits = re.sub(r"\s+", "", rest_number)

                if country_code:
                    full_number = f"+{country_code} {number_digits}"  # Add space after country code
                else:
                    full_number = number_digits

                print("Trying number:", full_number)  # Now shows: +33 3344335544

                try:
                    for region in ["US", "GB", "IN", "AU", "FR", "DE", "CN", "JP"]:
                        parsed_number = phonenumbers.parse(full_number, region)
                        if phonenumbers.is_valid_number(parsed_number):
                            formatted_number = phonenumbers.format_number(
                                parsed_number,
                                phonenumbers.PhoneNumberFormat.INTERNATIONAL
                            )
                            phone_numbers.add(formatted_number)
                            break
                except phonenumbers.NumberParseException as e:
                    print("Parse error:", e)
                    continue
                except Exception:
                    pass

            for token in doc:
                if token.pos_ == "NUM" and token.dep_ in {"nummod", "attr"}:
                    next_token = token.nbor(1) if token.i + 1 < len(doc) else None
                    if next_token and next_token.text.lower() in {"employees", "staff", "team", "personnel", "members",
                                                                  "people"}:
                        employee_counts.add(token.text)
                    elif re.match(r"^\d{2,7}\+?$", token.text) and int(token.text.rstrip("+")) >= 10:
                        employee_counts.add(token.text)

            return {
                "urls": urls,
                "phone_numbers": list(phone_numbers),
                "employee_count": list(employee_counts),
                "company_type": list(company_types)
            }
        except Exception as e:
            print(f"Error extracting employee count/url/phone: {str(e)}")
            return {"urls": [], "phone_numbers": [], "employee_count": [], "company_type": []}

    def extract_founding_year(text):
        try:
            patterns = [
                r"\bFounded\s*(?:in)?\s*(\d{4})\b",
                r"\bEstablished\s*(?:in)?\s*(\d{4})\b",
                r"\bEst\.\s*(\d{4})\b"
            ]
            years = []
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                years.extend(map(int, matches))
            return min(years) if years else None
        except Exception as e:
            print(f"Error extracting founding year: {str(e)}")
            return None

    def extract_headquarters(text):
        try:
            hq_match = re.search(
                r"\bHeadquarters\s*[:\-]?\s*([\s\S]+?)(?=\n(?:Website|Industry|Employees|Specialties|Founded|$))",
                text, re.IGNORECASE
            )
            address_match = re.search(
                r"\bAddress\s*[:\-]?\s*([\s\S]+?)(?=\n(?:Website|Email|Phone|$))",
                text, re.IGNORECASE
            )

            hq_result = hq_match.group(1).strip() if hq_match else None
            address_result = address_match.group(1).strip() if address_match else None

            return hq_result or address_result
        except Exception as e:
            print(f"Error extracting headquarters: {str(e)}")
            return None

    def extract_specialties(text):
        try:
            match = re.search(r"Specialties\s*(.*)", text, re.DOTALL)
            specialties_text = match.group(1).strip() if match else ""

            doc = nlp(specialties_text[:10000] if specialties_text else text[:10000])
            spacy_specialties = {chunk.text.strip() for chunk in doc.noun_chunks}

            if specialties_text:
                keybert_specialties = {
                    kw[0] for kw in kw_model.extract_keywords(
                        specialties_text, keyphrase_ngram_range=(1, 2),
                        stop_words="english", use_mmr=True, diversity=0.7, top_n=20
                    )
                }
            else:
                keybert_specialties = {
                    kw[0] for kw in kw_model.extract_keywords(
                        text[:5000], keyphrase_ngram_range=(1, 2),
                        stop_words="english", use_mmr=True, diversity=0.7, top_n=10
                    )
                }

            combined_specialties = spacy_specialties | keybert_specialties
            cleaned_specialties = [
                specialty for specialty in combined_specialties
                if specialty and not re.match(r'https?://|www\.', specialty)
                   and not any(keyword in specialty.lower() for keyword in ['url', 'http', 'com', 'company'])
                   and len(specialty) > 3
            ]
            return list(set(map(str.lower, sorted(cleaned_specialties, key=len, reverse=True))))
        except Exception as e:
            print(f"Error extracting specialties: {str(e)}")
            return []

    def extract_employee_count(text):
        try:
            regex_pattern = r"(\d{1,3}(?:,\d{3})*\+?)\s*(?:employees|members|staff|team|personnel|workers|people)"
            matches = re.findall(regex_pattern, text, re.IGNORECASE)
            employee_counts = []

            for match in matches:
                cleaned_match = re.sub(r"[^\d]", "", match)
                if cleaned_match:
                    try:
                        employee_counts.append(int(cleaned_match))
                    except ValueError:
                        pass

            return employee_counts
        except Exception as e:
            print(f"Error extracting employee count: {str(e)}")
            return []

    def extract_emails(text):
        try:
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, text)
            return list(set(emails))
        except Exception as e:
            print(f"Error extracting emails: {str(e)}")
            return []

    def extract_social_media(text):
        try:
            social_patterns = {
                'LinkedIn': r'linkedin\.com/(?:company|in)/([^/\s"\'>]+)',
                'Facebook': r'facebook\.com/([^/\s"\'>]+)',
                'Twitter': r'(?:twitter|x)\.com/([^/\s"\'>]+)',
                'Instagram': r'instagram\.com/([^/\s"\'>]+)'
            }

            social_matches = {}
            for platform, pattern in social_patterns.items():
                matches = re.findall(pattern, text)
                if matches:
                    social_matches[platform] = matches[0]

            return social_matches
        except Exception as e:
            print(f"Error extracting social media: {str(e)}")
            return {}

    try:
        # Use either the BERT or DeepSeek model for entity recognition based on parameter
        if use_deepseek:
            # Here you would use your DeepSeek model
            # results = deepseek_ner(text[:10000])
            results = ner_pipeline(text[:10000])  # Fallback to regular pipeline for now
            model_used = "DeepSeek"
        else:
            results = ner_pipeline(text[:10000])
            model_used = "BERT"

        results_final = filter_entities(results)
        additional_details = extract_employee_count_url_phone_pos(text)
        employee_count = extract_employee_count(text)
        headquarters = extract_headquarters(text)
        founding_year = extract_founding_year(text)
        specialties = extract_specialties(text)
        emails = extract_emails(text)
        social_media = extract_social_media(text)

        company_name = "Unknown Company"
        company_country = ""
        company_phone = ""
        company_size = ""
        company_industry = ""
        company_revenue = ""
        company_email = ""
        company_type = ""
        company_address = ""
        company_description = ""
        company_cities = []
        company_states = []

        # Track which fields came from which model
        extraction_sources = {
            "model_used": model_used
        }

        if results_final['organization']:
            company_name = results_final['organization'][0]
            extraction_sources["name"] = model_used

        company_name_match = re.search(r"Company Name:\s*([^\n]+)", text)
        if company_name_match:
            company_name = company_name_match.group(1).strip()
            extraction_sources["name"] = "Text pattern"

        if results_final['country']:
            company_country = results_final['country'][0]
            extraction_sources["country"] = model_used

        if results_final['city']:
            company_cities = results_final['city']
            extraction_sources["cities"] = model_used

        if results_final['state']:
            company_states = results_final['state']
            extraction_sources["states"] = model_used

        if additional_details['phone_numbers']:
            company_phone = additional_details['phone_numbers'][0]
            extraction_sources["phone"] = "Regex pattern"

        if emails:
            company_email = emails[0]
            extraction_sources["email"] = "Regex pattern"

        if additional_details['company_type']:
            company_type = additional_details['company_type'][0]
            extraction_sources["company_type"] = "Regex pattern"

        if employee_count:
            company_size = str(employee_count[0])
            extraction_sources["company_size"] = "Regex pattern"
        elif additional_details['employee_count']:
            company_size = additional_details['employee_count'][0]
            extraction_sources["company_size"] = "NLP pattern"

        industry_match = re.search(r"Industry:\s*([^\n]+)", text)
        if industry_match:
            company_industry = industry_match.group(1).strip()
            extraction_sources["industry"] = "Text pattern"
        elif specialties:
            company_industry = specialties[0]
            extraction_sources["industry"] = "KeyBERT extraction"

        description_match = re.search(r"Meta Description:\s*([^\n]+)", text)
        if description_match:
            company_description = description_match.group(1).strip()
            extraction_sources["description"] = "Meta tag"
        elif "About Content:" in text:
            about_content_match = re.search(r"About Content:\s*-\s*([^\n]+)", text)
            if about_content_match:
                company_description = about_content_match.group(1).strip()
                extraction_sources["description"] = "About page"

        revenue_match = re.search(r"Revenue:\s*([^\n]+)", text)
        if revenue_match:
            company_revenue = revenue_match.group(1).strip()
            extraction_sources["revenue"] = "Text pattern"
        else:
            revenue_patterns = [
                r"(?:annual|yearly)\s+revenue\s+(?:of\s+)?[$£€]?\s*(\d+(?:[.,]\d+)?\s*(?:million|billion|m|b|k)?)",
                r"revenue\s*[:-]\s*[$£€]?\s*(\d+(?:[.,]\d+)?\s*(?:million|billion|m|b|k)?)"
            ]

            for pattern in revenue_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    company_revenue = match.group(1).strip()
                    extraction_sources["revenue"] = "Regex pattern"
                    break

        # Create the company_info object with all the extracted data
        company_info = CompanyInfo2(
            name=company_name,
            industry=company_industry,
            headquarters=headquarters or "",
            company_size=company_size,
            website=additional_details['urls'][0] if additional_details['urls'] else "",
            country=company_country,
            phone_number=company_phone,
            revenue=company_revenue,
            email=company_email,
            social_media=social_media,
            cities=company_cities,
            states=company_states,
            founding_year=str(founding_year) if founding_year else "",
            specialties=specialties[:10] if specialties else [],
            description=company_description,
            company_type=company_type
        )

        return company_info

    except Exception as e:
        print(f"Error in extract_company_info: {str(e)}")
        # Return a default CompanyInfo object with minimal information
        return CompanyInfo2(
            name="Unknown Company",
            industry="",
            headquarters="",
            company_size="",
            website="",
            country="",
            phone_number="",
            revenue=""
        )


def serialize(obj):
    if hasattr(obj, '__dict__'):
        return {k: serialize(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [serialize(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    else:
        return obj
#result = ' '.join(results)

#print(llm_model(result))
#print(bert_model(result))
st.title("🔍 Website Analyzer: LLM vs BERT")

# Input box for website URL
#st.set_page_config(page_title="Website Analyzer", layout="wide")

# ✅ Everything else comes *after*
raw_url = st.text_input("Enter a website URL:", "baden-regio.ch")
nlp = spacy.load("en_core_web_sm")
custom_model = SentenceTransformer("all-mpnet-base-v2")
kw_model = KeyBERT(model=custom_model)

model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
ner_pipeline = pipeline("ner", model=model_name, tokenizer=model_name, aggregation_strategy="simple")
import sqlite3
from datetime import datetime

def save_result(url, llm_output, bert_output):
    conn = sqlite3.connect("results.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO results (url, llm_output, bert_output, timestamp)
        VALUES (?, ?, ?, ?)
    """, (url, llm_output, bert_output, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

    # Reset session state to refresh the company list in sidebar
    st.session_state.show_count = 10


# Initialize the DB
init_db()
all_results = get_all_results()
all_urls = list(set(r[0] for r in all_results))  # Unique company URLs
total_companies = len(all_urls)

# Display in sidebar
st.sidebar.markdown("## 🏢 Companies Processed")
st.sidebar.markdown(f"**Total Companies:** {total_companies}")

# Load more logic
if 'show_count' not in st.session_state:
    st.session_state.show_count = 10  # Start with 10

for company in all_urls[:st.session_state.show_count]:
    st.sidebar.markdown(f"- {company}")

if st.sidebar.button("🔄 Load More"):
    st.session_state.show_count += 10


#st.title("🔍 Website Analyzer: LLM vs BERT")

# Sidebar search
st.sidebar.header("🔎 Search Previous Results")
search_query = st.sidebar.text_input("Enter website (e.g., baden-regio.ch)")

if search_query:
    search_query = search_query.lower().strip()
    matches = [r for r in get_all_results() if search_query in r[0].lower()]

    st.sidebar.markdown("### Results Found:")
    if matches:
        for url, llm_output, bert_output, timestamp in matches:
            with st.sidebar.expander(f"{url} ({timestamp})"):
                st.markdown("**LLM Output:**")
                st.text(llm_output[:300] + "..." if len(llm_output) > 300 else llm_output)
                st.markdown("**BERT Output:**")
                st.text(bert_output[:300] + "..." if len(bert_output) > 300 else bert_output)
    else:
        st.sidebar.info("No results found for this website.")
if st.button("Analyze"):
    with st.spinner("Scraping and analyzing the website..."):
        try:
            chrome_options = Options()
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-notifications")
            chrome_options.add_argument("--disable-popup-blocking")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument(
                "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
            )

            driver = webdriver.Chrome(options=chrome_options)

            if '.' not in raw_url:
                st.subheader("Scraping output")
                llm_output=scrape_linkedin_company(raw_url)
                bert_output=llm_output
                st.write(llm_output)
                save_result(raw_url, json.dumps(serialize(llm_output)), json.dumps(serialize(bert_output)))


            else:
                normalized_url = normalize_url(raw_url)
                success, results = scrape_website(normalized_url, driver)

                if not success:
                    st.error("Failed to scrape the website.")

                else:
                    result_text = ' '.join(results)

                    # Layout for side-by-side results
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("🔮 LLM Model Output")
                        llm_output=llm_model(result_text)
                        st.write(llm_output)

                    with col2:
                        st.subheader("🧠 BERT Model Output")
                        bert_output=bert_model(result_text)
                        st.write(bert_output)
                    save_result(normalized_url, json.dumps(serialize(llm_output)), json.dumps(serialize(bert_output)))

        except Exception as e:
            st.error(f"An error occurred: {e}")




