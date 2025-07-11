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
from keybert import KeyBERT

app = FastAPI()



class CompanyInfo(BaseModel):
    name: str
    industry: str
    headquarters: str
    company_size: str
    website: str
    country: str
    phone_number: str
    revenue: str


def connect_to_mongodb():
    connection_string = "mongodb://b2b:b2bnova%2Ak@192.168.2.165:49153/?authSource=admin"
    db_name = "warehouse"
    collection_name = "10000_US_profiles"

    try:
        client = pymongo.MongoClient(connection_string)
        db = client[db_name]
        collection = db[collection_name]
        return collection, client
    except Exception as e:
        print(f"MongoDB connection error: {str(e)}")
        return None, None


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


def scrape_website(url, timeout=20):
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


def extract_company_info(scraped_text: str) -> dict:
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


def create_mongodb_update(existing_doc, company_info):
    update_data = {
        "extracted_info": {
            "name": company_info.name,
            "industry": company_info.industry,
            "headquarters": company_info.headquarters,
            "company_size": company_info.company_size,
            "website": company_info.website,
            "country": company_info.country,
            "phone_number": company_info.phone_number,
            "revenue": company_info.revenue,
            # Add additional fields that weren't in the original structure
            #"email": company_info.email,
            #"founding_year": company_info.founding_year,
            #"description": company_info.description,
            #"company_type": company_info.company_type
        }
    }

    for field in ["name", "industry", "headquarters", "website", "country", "phone_number", "revenue"]:
        value = getattr(company_info, field)
        if value:  # Only update if value exists
            # For fields that might already exist in the document
            if field in existing_doc and existing_doc[field] and existing_doc[field] != value:
                # Append new value with comma if it's different from existing
                update_data[field] = f"{existing_doc[field]}, {value}"
            else:
                # Simply set the value if field doesn't exist or is empty
                update_data[field] = value

    # Handle company_size separately
    if company_info.company_size:
        # Try to normalize to a range format if possible
        size_value = company_info.company_size
        if existing_doc.get("size") and existing_doc["size"] != size_value:
            update_data["size"] = f"{existing_doc['size']}, {size_value}"
        else:
            update_data["size"] = size_value

    # Handle potential new fields from extended extraction
    # Cities
    """
    if company_info.cities:
        new_cities = set(company_info.cities)
        existing_cities = set(existing_doc.get("cities", "").split(", ")) if existing_doc.get("cities") else set()
        all_cities = existing_cities.union(new_cities)
        update_data["cities"] = ", ".join(filter(None, all_cities))

    # States
    if company_info.states:
        new_states = set(company_info.states)
        existing_states = set(existing_doc.get("states", "").split(", ")) if existing_doc.get("states") else set()
        all_states = existing_states.union(new_states)
        update_data["states"] = ", ".join(filter(None, all_states))

    # Description
    if company_info.description:
        update_data["description"] = company_info.description

    # Email
    if company_info.email:
        if existing_doc.get("email") and existing_doc["email"] != company_info.email:
            update_data["email"] = f"{existing_doc['email']}, {company_info.email}"
        else:
            update_data["email"] = company_info.email

    # Company type
    if company_info.company_type:
        if existing_doc.get("type") and existing_doc["type"] != company_info.company_type:
            update_data["type"] = f"{existing_doc['type']}, {company_info.company_type}"
        else:
            update_data["type"] = company_info.company_type

    # Founding year
    if company_info.founding_year:
        update_data["founded"] = company_info.founding_year

    # Social media
    if company_info.social_media:
        for platform, handle in company_info.social_media.items():
            field_name = f"social_{platform.lower()}"
            update_data[field_name] = handle

    # Specialties/tags - store as array
    if company_info.specialties:
        existing_specialties = existing_doc.get("specialties", [])
        if not isinstance(existing_specialties, list):
            existing_specialties = []

        new_specialties = company_info.specialties
        all_specialties = list(set(existing_specialties + new_specialties))
        update_data["specialties"] = all_specialties
    """

    return update_data


@app.get("/website-scrape", response_model=CompanyInfo)
async def get_website_and_scrape(use_deepseek: bool = False):
    collection, client = None, None

    try:
        collection, client = connect_to_mongodb()
        if collection is None:
            raise HTTPException(status_code=500, detail="Failed to connect to MongoDB")

        cursor = collection.find({
            "website": {"$exists": True, "$ne": ""},
            "$or": [
                {"flag": {"$exists": False}},
                {"flag": False}
            ]
        })
        websites = list(cursor)

        if not websites:
            raise HTTPException(status_code=404, detail="No unscraped websites found in the database")

        processed_count = 0
        last_processed_company = None

        for website_doc in websites:
            doc_id = website_doc.get('_id')
            raw_url = website_doc.get('website', '')
            company_name = website_doc.get('name', 'Unknown Company')

            normalized_url = normalize_url(raw_url)
            if not normalized_url:
                continue

            try:
                success, scraped_data = scrape_website(normalized_url)

                existing_results = website_doc.get('scrape_results', '')

                if existing_results:
                    combined_results = f"{existing_results},\n\n{scraped_data}"
                else:
                    combined_results = scraped_data

                collection.update_one(
                    {"_id": doc_id},
                    {"$set": {
                        "last_scraped": datetime.now(),
                        "scrape_success": success,
                        "scrape_results": combined_results,
                        "flag": True
                    }}
                )

                if success:
                    company_info = extract_company_info(scraped_data)

                    if not company_info.name or company_info.name == "Unknown Company":
                        company_info.name = company_name

                    if not company_info.website:
                        company_info.website = normalized_url

                    # Create the update data with all fields
                    update_data = create_mongodb_update(website_doc, company_info)

                    # Update the document with all fields
                    collection.update_one(
                        {"_id": doc_id},
                        {"$set": update_data}
                    )

                    processed_count += 1
                    last_processed_company = company_info
            except Exception as scrape_error:
                print(f"Error scraping website {normalized_url}: {str(scrape_error)}")

                collection.update_one(
                    {"_id": doc_id},
                    {"$set": {"flag": True, "last_scraped": datetime.now()}}
                )
                continue

        if processed_count > 0:
            return last_processed_company
        else:
            raise HTTPException(status_code=404, detail="No valid websites could be scraped")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in API: {str(e)}")

    finally:
        if client:
            client.close()



