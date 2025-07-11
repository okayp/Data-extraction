from fastapi import FastAPI, HTTPException, Query
import time
import re
import os
import socket
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
from selenium.webdriver import ActionChains
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
from pydantic import BaseModel, Field
from langchain_ollama import OllamaLLM
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
import faiss
import numpy as np

try:
    import pyautogui  # For fallback translation method
except ImportError:
    print("PyAutoGUI not available - fallback translation method will be limited")

from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

app = FastAPI()

# LinkedIn credentials - you should store these securely in environment variables
LINKEDIN_EMAIL = os.environ.get("LINKEDIN_EMAIL", "your_email@example.com")
LINKEDIN_PASSWORD = os.environ.get("LINKEDIN_PASSWORD", "your_password")

try:
    nlp = spacy.load("en_core_web_sm")
    custom_model = SentenceTransformer("all-mpnet-base-v2")
    kw_model = KeyBERT(model=custom_model)

    model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
    ner_pipeline = pipeline("ner", model=model_name, tokenizer=model_name, aggregation_strategy="simple")

    # PLACEHOLDER FOR DEEPSEEK R1 MODEL INITIALIZATION
    # For this example, we're using the same BERT model as a placeholder
    #deepseek_model_name = "deepseek-ai/deepseek-ner-base"  # Replace with actual model name
    #deepseek_tokenizer = AutoTokenizer.from_pretrained(model_name)
    #deepseek_model = AutoModelForTokenClassification.from_pretrained(model_name)
    #deepseek_ner = pipeline("ner", model=deepseek_model, tokenizer=deepseek_tokenizer, aggregation_strategy="simple")
except Exception as e:
    print(f"Error initializing NLP components: {str(e)}")


class CompanyInfo(BaseModel):
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


class LinkedInData(BaseModel):
    company_name: Optional[str] = None
    description: Optional[str] = None
    industry: Optional[str] = None
    company_size: Optional[str] = None
    headquarters: Optional[str] = None
    website: Optional[str] = None
    specialties: Optional[List[str]] = []
    founded: Optional[str] = None
    url: Optional[str] = None


class DualModelResult(BaseModel):
    website_name: str
    bert_data: Optional[CompanyInfo] = None
    deepseek_data: Optional[CompanyInfo] = None
    linkedin_data: Optional[LinkedInData] = None


def connect_to_mongodb():
    connection_string = "mongodb://b2b:b2bnova%2Ak@192.168.2.165:49153/?authSource=admin"
    db_name = "warehouse"
    collection_name = "10000_US_profiles"
    new_collection_name = f"{collection_name}_new"

    try:
        client = pymongo.MongoClient(connection_string)
        db = client[db_name]
        collection = db[collection_name]

        # Check if the new collection exists
        if new_collection_name in db.list_collection_names():
            # Get the new collection
            new_collection = db[new_collection_name]

            # Check if the problematic index exists and drop it if it does
            indexes = new_collection.index_information()
            if 'source_id_1' in indexes:
                print("Dropping problematic source_id_1 index from new collection")
                new_collection.drop_index('source_id_1')

        else:
            # Create the new collection without the problematic index
            new_collection = db[new_collection_name]
            # Ensure we have a unique index only on website_name
            new_collection.create_index([("website_name", pymongo.ASCENDING)], unique=True)
            print(f"Created new collection {new_collection_name} with website_name index")

        return collection, new_collection, client
    except Exception as e:
        print(f"MongoDB connection error: {str(e)}")
        return None, None, None


def normalize_url(raw_url):
    if not raw_url:
        return None

    raw_url = str(raw_url).strip()
    if raw_url.lower() in ('data', 'null', 'n/a', 'none', ''):
        return None

    # Ensure scheme
    if not raw_url.startswith(('http://', 'https://')):
        raw_url = 'https://' + raw_url

    parsed = urlparse(raw_url)
    host = parsed.netloc.lower()

    # Try the host as-is
    try:
        socket.gethostbyname(host)
    except socket.gaierror:
        # Fallback: add www. if not already there
        if not host.startswith('www.'):
            www_host = 'www.' + host
            try:
                socket.gethostbyname(www_host)
                host = www_host  # Use www. version if it resolves
            except socket.gaierror:
                return None  # Neither version resolves

    # Rebuild the URL
    scheme = parsed.scheme or 'https'
    path = parsed.path or ''
    query = '?' + parsed.query if parsed.query else ''
    fragment = '#' + parsed.fragment if parsed.fragment else ''

    return f"{scheme}://{host}{path}{query}{fragment}"


def scrape_website(url, timeout=20):
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--lang=en")  # Request English language
    chrome_options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
    )

    # Add translation preferences to automatically translate to English
    prefs = {
        "translate_whitelists": {},  # We'll set this to translate from any language to English
        "translate": {"enabled": "true"}
    }

    # Create whitelist for common languages
    common_languages = ["fr", "es", "de", "it", "pt", "nl", "ru", "zh", "ja", "ko", "ar", "hi", "tr"]
    for lang in common_languages:
        prefs["translate_whitelists"][lang] = "en"

    chrome_options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(options=chrome_options)
    results = []
    results.append(f"Website: {url}")
    results.append(f"Scrape Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    driver.set_page_load_timeout(timeout)

    try:
        driver.get(url)
        time.sleep(5)

        # Check if translation was offered and accept it if needed
        try:
            # Look for Google Translate bar elements
            translation_elements = [
                "//div[contains(@class, 'translation-prompt')]//button",
                "//div[contains(@id, 'translate')]//button[contains(text(), 'Translate')]",
                "//span[contains(text(), 'Translate this page')]",
                "//button[contains(text(), 'Translate')]"
            ]

            for xpath in translation_elements:
                try:
                    translate_button = WebDriverWait(driver, 3).until(
                        EC.element_to_be_clickable((By.XPATH, xpath))
                    )
                    translate_button.click()
                    time.sleep(2)  # Wait for translation to complete
                    break
                except:
                    continue

            # Manual right-click approach as fallback
            if driver.page_source.find("html lang=\"en\"") == -1:
                # If page is not in English, try context menu translation
                actions = ActionChains(driver)
                actions.context_click(driver.find_element(By.TAG_NAME, "body")).perform()
                time.sleep(1)

                # Try to navigate to "Translate to English" option using keyboard
                # This part requires PyAutoGUI which should be imported
                try:
                    import pyautogui
                    for _ in range(3):  # Adjust based on menu position
                        pyautogui.press('down')
                    pyautogui.press('right')  # Move to submenu
                    pyautogui.press('enter')  # Select English
                    time.sleep(3)  # Wait for translation
                except ImportError:
                    results.append("Note: PyAutoGUI not available for fallback translation")

        except Exception as e:
            results.append(f"Translation handling note: {str(e)}")

        # Handle cookie acceptance as in original code
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

            button_texts = ["Accept", "Accept All", "Akzeptieren", "Accepteren", "Aceptar", "Accepter"]
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

        # Continue with the rest of your scraping code as before
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        text = soup.get_text()

        # Rest of your existing scraping code continues here...
        results.append("\n--- HOME PAGE INFORMATION ---")

        title_element = soup.find('title')
        if title_element:
            company_name = re.sub(r'\s*[-|]\s*.*$', '', title_element.text).strip()
            results.append(f"Company Name: {company_name}")

        # Try to get meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            results.append(f"Meta Description: {meta_desc.get('content', '')}")

        # Try to visit About page if available
        about_link = None
        for link in soup.find_all('a', href=True):
            link_text = link.text.lower()
            if 'about' in link_text and ('us' in link_text or 'company' in link_text):
                about_link = link['href']
                break

        if about_link:
            # Handle relative URLs
            if about_link.startswith('/'):
                parsed_url = urlparse(url)
                about_link = f"{parsed_url.scheme}://{parsed_url.netloc}{about_link}"
            elif not about_link.startswith(('http://', 'https://')):
                # Handle case where URL doesn't have leading slash
                parsed_url = urlparse(url)
                about_link = f"{parsed_url.scheme}://{parsed_url.netloc}/{about_link}"

            try:
                # Visit About page
                driver.get(about_link)
                time.sleep(3)
                about_page = driver.page_source
                about_soup = BeautifulSoup(about_page, 'html.parser')

                # Add About page content
                results.append("\n--- ABOUT PAGE CONTENT ---")

                # Extract key paragraphs
                paragraphs = about_soup.find_all('p')
                for i, p in enumerate(paragraphs[:5]):  # Get first 5 paragraphs
                    if len(p.text.strip()) > 50:  # Only substantial paragraphs
                        results.append(f"About Content - {i + 1}: {p.text.strip()}")
            except Exception as about_err:
                results.append(f"Error accessing About page: {str(about_err)}")

        full_text = "\n".join(results)

        driver.quit()

        return True, full_text

    except TimeoutException:
        error_msg = "Website loading timed out"
        results.append(f"\nError: {error_msg}")
        driver.quit()
        return False, "\n".join(results)

    except WebDriverException as e:
        error_msg = f"WebDriver exception - {str(e)}"
        results.append(f"\nError: {error_msg}")
        driver.quit()
        return False, "\n".join(results)

    except Exception as e:
        error_msg = f"Error during scraping: {str(e)}"
        results.append(f"\nError: {error_msg}")
        driver.quit()
        return False, "\n".join(results)


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
                    "company_name": null,
                    "description": null,
                    "industry": null,
                    "company_size": null,
                    "headquarters": null,
                    "website": null,
                    "specialties": [],
                    "founded": null,
                    "url": window.location.href
                };

                // Extract company name (usually in h1)
                const nameElements = document.querySelectorAll('h1');
                if (nameElements.length > 0) {
                    data["company_name"] = nameElements[0].textContent.trim();
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
                    ["industry", "Industry"],
                    ["company_size", "Company size"],
                    ["headquarters", "Headquarters"],
                    ["founded", "Founded"],
                    ["website", "Website"]
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
                    data["description"] = descElements[0].textContent.trim();
                } else {
                    // Method 2: Look for paragraphs in the about page
                    const paragraphs = document.querySelectorAll('p');
                    for (const p of paragraphs) {
                        const text = p.textContent.trim();
                        if (text.length > 50) { // Likely a description if it's a longer paragraph
                            data["description"] = text;
                            break;
                        }
                    }
                }

                // Special handling for specialties (comma-separated)
                const specialtiesText = findFieldByLabel("Specialties");
                if (specialtiesText) {
                    data["specialties"] = specialtiesText.split(',')
                        .map(s => s.trim())
                        .filter(s => s.length > 0);
                }

                return data;
            }

            return extractCompanyBasics();
        """)

        # Print extracted data
        print(f"\n--- Extracted LinkedIn Data for {display_name} ---")
        for key, value in company_data.items():
            if key != "specialties":
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

class CompanyInfo2(BaseModel):
    name: Optional[Union[str, List[str]]] = Field(default=None,description="Company name")
    industry: Optional[Union[str, List[str]]] = Field(default=None,description="Company industry")
    headquarters: Optional[Union[str, List[str]]] = Field(default=None,description="Headquarters location")
    company_size: Optional[Union[str, List[str]]] = Field(default=None,description="Number of employees")
    website:Optional[Union[str,List[str]]]=Field(default=None,description="Give the website of the company")
    location:Optional[Union[str,List[str]]]=Field(default=None,description='Give the city, state and country of the company')



    phone_number:Optional[Union[str,List[str]]]=Field(default=None,description="Give the phone number of company if  available")
    revenue:Optional[Union[str,List[str]]]=Field(default=None,description="Give the revenue of the company if mentioned")
    extraction_model: Optional[str] = None

def extract_using_deepseek(text):
    query = text
    knowledge_base = [text]

    model = SentenceTransformer('all-MiniLM-L6-v2')
    kb_embeddings = model.encode(knowledge_base)

    index = faiss.IndexFlatL2(kb_embeddings.shape[1])
    index.add(np.array(kb_embeddings))

    query_embedding = model.encode([query])

    D, I = index.search(query_embedding, k=2)
    retrieved_docs = [knowledge_base[i] for i in I[0]]

    llm = OllamaLLM(model="deepseek-r1:7b")
    parser = PydanticOutputParser(pydantic_object=CompanyInfo2)

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
                    "location": "Give the location of the company in this order: city, state, country"
                    "phone_number": "Company Phone Number (if available, otherwise empty string)",
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


def extract_company_info(text, use_deepseek=False, debug=False):
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

                # Changed from 0.9 to 0.99 as per second code
                if entity_group == 'ORG' and entity_score > 0.99:
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

            # Enhanced phone pattern from second code
            phone_pattern = r"(?:\+?(\d{1,3})[ ]?)?([\d\s]{10,15})"
            raw_phone_numbers = re.findall(phone_pattern, text)

            for match in raw_phone_numbers:
                country_code, rest_number = match
                number_digits = re.sub(r"\s+", "", rest_number)

                if country_code:
                    full_number = f"+{country_code} {number_digits}"  # Add space after country code
                else:
                    full_number = number_digits

                if debug:
                    print(f"Trying number: {full_number}")

                try:
                    for region in ["US", "GB", "IN", "AU", "FR", "DE", "CN", "JP", "TR"]:
                        try:
                            parsed_number = phonenumbers.parse(full_number, region)
                            if phonenumbers.is_valid_number(parsed_number):
                                formatted_number = phonenumbers.format_number(
                                    parsed_number,
                                    phonenumbers.PhoneNumberFormat.INTERNATIONAL
                                )
                                phone_numbers.add(formatted_number)
                                break
                        except Exception as e:
                            if debug:
                                print(f"Parse error: {str(e)}")
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
            # Using improved cleaning logic from second code to better remove junk patterns
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
            # Here we're using the same BERT model as a placeholder for DeepSeek
            results = deepseek_ner(text[:10000])
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

        # Create the company_info object with all the extracted data and track model used
        company_info = CompanyInfo(
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
            company_type=company_type,
            extraction_model=model_used  # Add which model was used
        )

        return company_info

    except Exception as e:
        print(f"Error in extract_company_info: {str(e)}")
        # Return a default CompanyInfo object with minimal information
        return CompanyInfo(
            name="Unknown Company",
            industry="",
            headquarters="",
            company_size="",
            website="",
            country="",
            phone_number="",
            revenue="",
            extraction_model="Error"  # Include error indication
        )


@app.get("/website-scrape", response_model=DualModelResult)
async def get_website_and_scrape(
        linkedin_company: str = Query(None, description="LinkedIn company name for additional scraping")):
    """
    Scrape a website from MongoDB collection and process it with both BERT and DeepSeek models.
    Optionally scrape LinkedIn if a company name is provided.
    Store results in a new collection with only the required fields.
    """
    collection, new_collection, client = None, None, None

    try:
        collection, new_collection, client = connect_to_mongodb()
        if collection is None or new_collection is None:
            raise HTTPException(status_code=500, detail="Failed to connect to MongoDB")

        # Find websites that need processing
        filter_condition = {
            "website": {"$exists": True, "$ne": ""}
        }

        cursor = collection.find(filter_condition)
        websites = list(cursor)

        if not websites:
            raise HTTPException(status_code=404, detail="No websites found for processing")

        linkedin_data = None
        if linkedin_company:
            try:
                print(f"Fetching LinkedIn data for {linkedin_company}")
                # Try to get data from LinkedIn
                linkedin_results = scrape_linkedin_company(
                    company_name=linkedin_company,
                    display_name=linkedin_company.capitalize(),
                    email=LINKEDIN_EMAIL,
                    password=LINKEDIN_PASSWORD
                )

                # Convert to our LinkedInData model
                linkedin_data = LinkedInData(
                    company_name=linkedin_results.get("company_name"),
                    description=linkedin_results.get("description"),
                    industry=linkedin_results.get("industry"),
                    company_size=linkedin_results.get("company_size"),
                    headquarters=linkedin_results.get("headquarters"),
                    website=linkedin_results.get("website"),
                    specialties=linkedin_results.get("specialties", []),
                    founded=linkedin_results.get("founded"),
                    url=linkedin_results.get("url")
                )
                print(f"Successfully fetched LinkedIn data for {linkedin_company}")
            except Exception as linkedin_err:
                print(f"Error fetching LinkedIn data: {str(linkedin_err)}")
                linkedin_data = None

        for website_doc in websites:
            raw_url = website_doc.get('website', '')
            company_name = website_doc.get('name', 'Unknown Company')

            normalized_url = normalize_url(raw_url)
            if not normalized_url:
                print(f"Skipping invalid URL that cannot be resolved: {raw_url}")
                continue

            try:
                # Check if we've already processed this website
                existing_in_new = new_collection.find_one({"website_name": normalized_url})
                if existing_in_new:
                    # If we're adding LinkedIn data to an existing entry
                    if linkedin_data and existing_in_new.get("linkedin_data") is None:
                        print(f"Website {normalized_url} already processed. Adding LinkedIn data...")
                        new_collection.update_one(
                            {"website_name": normalized_url},
                            {"$set": {"linkedin_data": linkedin_data.dict()}}
                        )

                        return DualModelResult(
                            website_name=normalized_url,
                            bert_data=CompanyInfo(**existing_in_new.get("bert_data", {})),
                            deepseek_data=CompanyInfo(**existing_in_new.get("deepseek_data", {})),
                            linkedin_data=linkedin_data
                        )
                    else:
                        print(f"Website {normalized_url} already processed. Skipping...")
                        continue

                print(f"Processing website: {normalized_url}")

                # Scrape the website using the original scraping function
                success, scraped_data = scrape_website(normalized_url)

                if not success:
                    print(f"Failed to scrape website: {normalized_url}")
                    # Store the failure with the minimal required fields
                    failure_doc = {
                        "website_name": normalized_url,
                        "bert_data": {
                            "name": company_name,
                            "industry": "",
                            "headquarters": "",
                            "company_size": "",
                            "website": normalized_url,
                            "country": "",
                            "phone_number": "",
                            "revenue": "",
                            "email": "",
                            "social_media": {},
                            "cities": [],
                            "states": [],
                            "founding_year": "",
                            "specialties": [],
                            "description": "",
                            "company_type": "",
                            "extraction_model": "BERT (Failed)"
                        },
                        "deepseek_data": {
                            "name": company_name,
                            "industry": "",
                            "headquarters": "",
                            "company_size": "",
                            "website": normalized_url,
                            "country": "",
                            "phone_number": "",
                            "revenue": "",
                            "email": "",
                            "social_media": {},
                            "cities": [],
                            "states": [],
                            "founding_year": "",
                            "specialties": [],
                            "description": "",
                            "company_type": "",
                            "extraction_model": "DeepSeek (Failed)"
                        },
                        "scrape_success_bert": False,
                        "scrape_success_deepseek": False
                    }

                    # Add LinkedIn data if available
                    if linkedin_data:
                        failure_doc["linkedin_data"] = linkedin_data.dict()

                    try:
                        new_collection.insert_one(failure_doc)
                    except pymongo.errors.DuplicateKeyError:
                        new_collection.update_one(
                            {"website_name": normalized_url},
                            {"$set": failure_doc}
                        )
                    continue

                # Process with BERT model
                print(f"Successfully scraped {normalized_url}, processing with BERT model...")
                bert_info = extract_company_info(scraped_data, use_deepseek=False)

                # Process with DeepSeek model
                print(f"Processing with DeepSeek model...")
                deepseek_info = extract_using_deepseek(scraped_data)

                # Ensure company names are set
                if not bert_info.name or bert_info.name == "Unknown Company":
                    bert_info.name = company_name

                if not deepseek_info.name or deepseek_info.name == "Unknown Company":
                    deepseek_info.name = company_name

                # Ensure websites are set
                if not bert_info.website:
                    bert_info.website = normalized_url

                if not deepseek_info.website:
                    deepseek_info.website = normalized_url

                # Create document with only the required fields
                result_doc = {
                    "website_name": normalized_url,
                    "bert_data": bert_info.dict(),  # Store the complete BERT results
                    "deepseek_data": deepseek_info.dict(),  # Store the complete DeepSeek results
                    "scrape_success_bert": True,
                    "scrape_success_deepseek": True
                }

                # Add LinkedIn data if available
                if linkedin_data:
                    result_doc["linkedin_data"] = linkedin_data.dict()

                # Insert or update the document
                try:
                    new_collection.insert_one(result_doc)
                    print(f"Successfully inserted data for {normalized_url}")
                except pymongo.errors.DuplicateKeyError:
                    new_collection.update_one(
                        {"website_name": normalized_url},
                        {"$set": result_doc}
                    )

                # Return the result for the first successfully processed website
                return DualModelResult(
                    website_name=normalized_url,
                    bert_data=bert_info,
                    deepseek_data=deepseek_info,
                    linkedin_data=linkedin_data
                )

            except Exception as e:
                print(f"Error processing website {normalized_url}: {str(e)}")
                traceback.print_exc()
                continue

        # If we get here, no websites were successfully processed
        raise HTTPException(status_code=404, detail="No websites could be successfully processed")

    except Exception as e:
        print(f"API Error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error in API: {str(e)}")

    finally:
        if client:
            client.close()


