"""
Web Scraper Agent - Performs web scraping and data extraction on any topic
"""
import asyncio
import aiohttp
import json
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from urllib.parse import urljoin, urlparse
import sqlite3

try:
    from bs4 import BeautifulSoup
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
except ImportError:
    BeautifulSoup = None
    webdriver = None

from .base_agent import BaseAgent, AgentTask
from .utils import sanitize_input, generate_task_id, format_timestamp, extract_urls, global_rate_limiter
from config import config

class ScrapingMethod(Enum):
    """Web scraping methods"""
    REQUESTS = "requests"
    SELENIUM = "selenium"
    API = "api"

class DataFormat(Enum):
    """Output data formats"""
    JSON = "json"
    CSV = "csv"
    TEXT = "text"
    HTML = "html"

class ScrapingStatus(Enum):
    """Scraping job status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ScrapingJob:
    """Represents a web scraping job"""
    id: str
    url: str
    target_data: List[str]  # What data to extract
    method: ScrapingMethod
    status: ScrapingStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ScrapedData:
    """Represents scraped data"""
    id: str
    job_id: str
    url: str
    data_type: str
    content: Dict[str, Any]
    scraped_at: datetime
    
    def __post_init__(self):
        if not isinstance(self.content, dict):
            self.content = {"raw_content": str(self.content)}

class WebScraperAgent(BaseAgent):
    """Web Scraper Agent for data extraction from websites"""
    
    def __init__(self):
        super().__init__(
            name="web_scraper",
            description="Performs web scraping and data extraction on any topic"
        )
        
        self.scraping_jobs: List[ScrapingJob] = []
        self.scraped_data: List[ScrapedData] = []
        self.db_path = "web_scraper.db"
        
        # Initialize database
        self._init_database()
        
        # Load existing data
        self._load_data()
        
        # Scraping configuration
        self.max_concurrent_requests = config.max_concurrent_requests
        self.scraping_delay = config.scraping_delay
        self.selenium_driver_path = config.selenium_driver_path
        
        # Common selectors for different types of content
        self.common_selectors = {
            "title": ["h1", "title", ".title", "#title"],
            "content": ["article", ".content", ".post-content", "main", ".article-body"],
            "links": ["a[href]"],
            "images": ["img[src]"],
            "text": ["p", ".text", ".description"],
            "price": [".price", ".cost", "[data-price]", ".amount"],
            "date": [".date", ".timestamp", "time", "[datetime]"],
            "author": [".author", ".by", ".writer", "[data-author]"]
        }
    
    def _init_database(self):
        """Initialize SQLite database for scraping data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create scraping jobs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scraping_jobs (
                    id TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    target_data TEXT NOT NULL,
                    method TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    results TEXT,
                    error_message TEXT,
                    metadata TEXT
                )
            ''')
            
            # Create scraped data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scraped_data (
                    id TEXT PRIMARY KEY,
                    job_id TEXT,
                    url TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    scraped_at TEXT NOT NULL,
                    FOREIGN KEY (job_id) REFERENCES scraping_jobs (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Web scraper database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize scraper database: {str(e)}")
    
    def _load_data(self):
        """Load scraping jobs and data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load scraping jobs
            cursor.execute("SELECT * FROM scraping_jobs")
            for row in cursor.fetchall():
                job = ScrapingJob(
                    id=row[0],
                    url=row[1],
                    target_data=json.loads(row[2]),
                    method=ScrapingMethod(row[3]),
                    status=ScrapingStatus(row[4]),
                    created_at=datetime.fromisoformat(row[5]),
                    completed_at=datetime.fromisoformat(row[6]) if row[6] else None,
                    results=json.loads(row[7]) if row[7] else None,
                    error_message=row[8],
                    metadata=json.loads(row[9]) if row[9] else {}
                )
                self.scraping_jobs.append(job)
            
            # Load scraped data
            cursor.execute("SELECT * FROM scraped_data")
            for row in cursor.fetchall():
                data = ScrapedData(
                    id=row[0],
                    job_id=row[1],
                    url=row[2],
                    data_type=row[3],
                    content=json.loads(row[4]),
                    scraped_at=datetime.fromisoformat(row[5])
                )
                self.scraped_data.append(data)
            
            conn.close()
            self.logger.info(f"Loaded {len(self.scraping_jobs)} jobs and {len(self.scraped_data)} data entries")
            
        except Exception as e:
            self.logger.error(f"Failed to load scraper data: {str(e)}")
    
    async def _process_task_impl(self, task: AgentTask) -> str:
        """Process web scraper tasks"""
        task_type = task.task_type.lower()
        content = sanitize_input(task.content)
        
        if task_type == "scrape_url":
            return await self._scrape_url(content, task.metadata)
        elif task_type == "scrape_search_results":
            return await self._scrape_search_results(content, task.metadata)
        elif task_type == "extract_data":
            return await self._extract_specific_data(content, task.metadata)
        elif task_type == "monitor_website":
            return await self._monitor_website(content, task.metadata)
        elif task_type == "scrape_social_media":
            return await self._scrape_social_media(content, task.metadata)
        elif task_type == "get_job_status":
            return await self._get_job_status(content, task.metadata)
        elif task_type == "list_jobs":
            return await self._list_scraping_jobs()
        elif task_type == "export_data":
            return await self._export_scraped_data(content, task.metadata)
        elif task_type == "health_check":
            return "Web Scraper Agent is ready to extract data from any website!"
        else:
            return await self._scrape_url(content, task.metadata)
    
    async def _scrape_url(self, url: str, metadata: Dict[str, Any]) -> str:
        """Scrape data from a specific URL"""
        try:
            # Validate URL
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            target_data = metadata.get("target_data", ["title", "content", "links"])
            method = metadata.get("method", "requests")
            output_format = metadata.get("format", "text")
            
            try:
                scraping_method = ScrapingMethod(method)
            except ValueError:
                scraping_method = ScrapingMethod.REQUESTS
            
            # Create scraping job
            job = ScrapingJob(
                id=generate_task_id(),
                url=url,
                target_data=target_data,
                method=scraping_method,
                status=ScrapingStatus.PENDING,
                created_at=datetime.now(),
                metadata=metadata
            )
            
            # Save job
            self._save_scraping_job(job)
            self.scraping_jobs.append(job)
            
            # Execute scraping
            job.status = ScrapingStatus.RUNNING
            self._save_scraping_job(job)
            
            scraped_results = await self._execute_scraping(job)
            
            # Update job with results
            job.status = ScrapingStatus.COMPLETED
            job.completed_at = datetime.now()
            job.results = scraped_results
            self._save_scraping_job(job)
            
            # Save scraped data
            for data_type, content in scraped_results.items():
                if content:  # Only save non-empty content
                    scraped_data = ScrapedData(
                        id=generate_task_id(),
                        job_id=job.id,
                        url=url,
                        data_type=data_type,
                        content=content if isinstance(content, dict) else {"data": content},
                        scraped_at=datetime.now()
                    )
                    self._save_scraped_data(scraped_data)
                    self.scraped_data.append(scraped_data)
            
            # Format results
            result = f"Web Scraping Results\n\n"
            result += f"**URL:** {url}\n"
            result += f"**Job ID:** {job.id}\n"
            result += f"**Method:** {scraping_method.value.title()}\n"
            result += f"**Target Data:** {', '.join(target_data)}\n"
            result += f"**Completed:** {format_timestamp(job.completed_at)}\n\n"
            
            # Display results based on format
            if output_format == "json":
                result += f"**Results (JSON):**\n```json\n{json.dumps(scraped_results, indent=2)}\n```"
            else:
                for data_type, content in scraped_results.items():
                    if content:
                        result += f"**{data_type.title()}:**\n"
                        if isinstance(content, list):
                            for item in content[:5]:  # Show first 5 items
                                result += f"• {str(item)[:100]}...\n"
                            if len(content) > 5:
                                result += f"... and {len(content) - 5} more items\n"
                        elif isinstance(content, dict):
                            for key, value in list(content.items())[:3]:  # Show first 3 items
                                result += f"• {key}: {str(value)[:100]}...\n"
                        else:
                            result += f"{str(content)[:500]}...\n"
                        result += "\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to scrape URL: {str(e)}")
            
            # Update job with error
            if 'job' in locals():
                job.status = ScrapingStatus.FAILED
                job.error_message = str(e)
                job.completed_at = datetime.now()
                self._save_scraping_job(job)
            
            return f"Sorry, I couldn't scrape the URL. Error: {str(e)}"
    
    async def _scrape_search_results(self, query: str, metadata: Dict[str, Any]) -> str:
        """Scrape search results for a query"""
        try:
            search_engine = metadata.get("search_engine", "google")
            max_results = metadata.get("max_results", 10)
            
            # Construct search URL
            search_urls = {
                "google": f"https://www.google.com/search?q={query.replace(' ', '+')}&num={max_results}",
                "bing": f"https://www.bing.com/search?q={query.replace(' ', '+')}&count={max_results}",
                "duckduckgo": f"https://duckduckgo.com/?q={query.replace(' ', '+')}"
            }
            
            search_url = search_urls.get(search_engine, search_urls["google"])
            
            # Create scraping job for search results
            job = ScrapingJob(
                id=generate_task_id(),
                url=search_url,
                target_data=["search_results"],
                method=ScrapingMethod.REQUESTS,
                status=ScrapingStatus.RUNNING,
                created_at=datetime.now(),
                metadata={"query": query, "search_engine": search_engine}
            )
            
            self._save_scraping_job(job)
            self.scraping_jobs.append(job)
            
            # Execute search scraping
            search_results = await self._scrape_search_engine(search_url, search_engine, max_results)
            
            # Update job
            job.status = ScrapingStatus.COMPLETED
            job.completed_at = datetime.now()
            job.results = {"search_results": search_results}
            self._save_scraping_job(job)
            
            result = f"Search Results for: {query}\n\n"
            result += f"**Search Engine:** {search_engine.title()}\n"
            result += f"**Results Found:** {len(search_results)}\n"
            result += f"**Job ID:** {job.id}\n\n"
            
            for i, item in enumerate(search_results[:10], 1):
                result += f"**{i}. {item.get('title', 'No Title')}**\n"
                result += f"   URL: {item.get('url', 'No URL')}\n"
                if item.get('description'):
                    result += f"   Description: {item['description'][:150]}...\n"
                result += "\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to scrape search results: {str(e)}")
            return f"Sorry, I couldn't scrape search results. Error: {str(e)}"
    
    async def _extract_specific_data(self, description: str, metadata: Dict[str, Any]) -> str:
        """Extract specific data types from websites"""
        try:
            urls = metadata.get("urls", [])
            data_types = metadata.get("data_types", ["text"])
            
            if not urls:
                # Try to extract URLs from description
                urls = extract_urls(description)
                if not urls:
                    return "Please provide URLs to scrape data from."
            
            results = {}
            
            for url in urls[:5]:  # Limit to 5 URLs
                try:
                    # Create individual job for each URL
                    job = ScrapingJob(
                        id=generate_task_id(),
                        url=url,
                        target_data=data_types,
                        method=ScrapingMethod.REQUESTS,
                        status=ScrapingStatus.RUNNING,
                        created_at=datetime.now()
                    )
                    
                    scraped_data = await self._execute_scraping(job)
                    results[url] = scraped_data
                    
                    job.status = ScrapingStatus.COMPLETED
                    job.completed_at = datetime.now()
                    job.results = scraped_data
                    
                    self._save_scraping_job(job)
                    self.scraping_jobs.append(job)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to scrape {url}: {str(e)}")
                    results[url] = {"error": str(e)}
            
            # Format results
            result = f"Data Extraction Results\n\n"
            result += f"**Description:** {description}\n"
            result += f"**Data Types:** {', '.join(data_types)}\n"
            result += f"**URLs Processed:** {len(results)}\n\n"
            
            for url, data in results.items():
                result += f"**{url}:**\n"
                if "error" in data:
                    result += f"   Error: {data['error']}\n"
                else:
                    for data_type, content in data.items():
                        if content:
                            result += f"   {data_type.title()}: {str(content)[:200]}...\n"
                result += "\n"
            
            return result

        except Exception as e:
            self.logger.error(f"Failed to extract specific data: {str(e)}")
            return f"Sorry, I couldn't extract the data. Error: {str(e)}"

    async def _monitor_website(self, url: str, metadata: Dict[str, Any]) -> str:
        """Monitor a website for changes"""
        try:
            check_interval = metadata.get("interval_hours", 24)
            target_elements = metadata.get("target_elements", ["content"])

            # This would typically set up a monitoring job
            # For now, we'll do a one-time check and explain the monitoring concept

            result = f"Website Monitoring Setup\n\n"
            result += f"**URL:** {url}\n"
            result += f"**Check Interval:** Every {check_interval} hours\n"
            result += f"**Monitoring:** {', '.join(target_elements)}\n\n"

            # Perform initial scrape
            job = ScrapingJob(
                id=generate_task_id(),
                url=url,
                target_data=target_elements,
                method=ScrapingMethod.REQUESTS,
                status=ScrapingStatus.RUNNING,
                created_at=datetime.now(),
                metadata={"monitoring": True, "interval": check_interval}
            )

            initial_data = await self._execute_scraping(job)

            job.status = ScrapingStatus.COMPLETED
            job.completed_at = datetime.now()
            job.results = initial_data

            self._save_scraping_job(job)
            self.scraping_jobs.append(job)

            result += f"**Initial Baseline Captured:**\n"
            for element, content in initial_data.items():
                if content:
                    result += f"• {element.title()}: {str(content)[:100]}...\n"

            result += f"\n**Note:** In a production environment, this would set up automated monitoring to check for changes every {check_interval} hours and notify you of any updates."

            return result

        except Exception as e:
            self.logger.error(f"Failed to set up monitoring: {str(e)}")
            return f"Sorry, I couldn't set up website monitoring. Error: {str(e)}"

    async def _scrape_social_media(self, platform_info: str, metadata: Dict[str, Any]) -> str:
        """Scrape social media content (with limitations)"""
        try:
            platform = metadata.get("platform", "twitter")
            content_type = metadata.get("content_type", "posts")

            result = f"Social Media Scraping Request\n\n"
            result += f"**Platform:** {platform.title()}\n"
            result += f"**Content Type:** {content_type}\n"
            result += f"**Query:** {platform_info}\n\n"

            result += "**Important Notice:**\n"
            result += "Social media scraping is subject to platform terms of service and rate limits. "
            result += "Many platforms require API access for legitimate data collection.\n\n"

            result += "**Recommended Approach:**\n"
            result += f"1. Use {platform.title()}'s official API\n"
            result += "2. Respect rate limits and terms of service\n"
            result += "3. Consider using authorized third-party tools\n"
            result += "4. Focus on publicly available content only\n\n"

            result += "**Alternative Solutions:**\n"
            result += "• Twitter API v2 for Twitter data\n"
            result += "• Facebook Graph API for Facebook content\n"
            result += "• Instagram Basic Display API for Instagram\n"
            result += "• LinkedIn API for professional content\n"

            return result

        except Exception as e:
            self.logger.error(f"Failed to process social media request: {str(e)}")
            return f"Sorry, I couldn't process the social media scraping request. Error: {str(e)}"

    async def _get_job_status(self, job_id: str, metadata: Dict[str, Any]) -> str:
        """Get status of a scraping job"""
        try:
            job = None
            for j in self.scraping_jobs:
                if j.id == job_id:
                    job = j
                    break

            if not job:
                return f"Scraping job '{job_id}' not found."

            result = f"Scraping Job Status\n\n"
            result += f"**Job ID:** {job.id}\n"
            result += f"**URL:** {job.url}\n"
            result += f"**Status:** {job.status.value.title()}\n"
            result += f"**Method:** {job.method.value.title()}\n"
            result += f"**Target Data:** {', '.join(job.target_data)}\n"
            result += f"**Created:** {format_timestamp(job.created_at)}\n"

            if job.completed_at:
                result += f"**Completed:** {format_timestamp(job.completed_at)}\n"
                duration = job.completed_at - job.created_at
                result += f"**Duration:** {duration.total_seconds():.2f} seconds\n"

            if job.error_message:
                result += f"**Error:** {job.error_message}\n"

            if job.results:
                result += f"\n**Results Summary:**\n"
                for data_type, content in job.results.items():
                    if content:
                        if isinstance(content, list):
                            result += f"• {data_type.title()}: {len(content)} items\n"
                        elif isinstance(content, dict):
                            result += f"• {data_type.title()}: {len(content)} fields\n"
                        else:
                            result += f"• {data_type.title()}: {len(str(content))} characters\n"

            return result

        except Exception as e:
            self.logger.error(f"Failed to get job status: {str(e)}")
            return f"Sorry, I couldn't get the job status. Error: {str(e)}"

    async def _list_scraping_jobs(self) -> str:
        """List all scraping jobs"""
        if not self.scraping_jobs:
            return "No scraping jobs found."

        result = "Scraping Jobs\n\n"

        # Group by status
        jobs_by_status = {}
        for job in self.scraping_jobs:
            status = job.status.value
            if status not in jobs_by_status:
                jobs_by_status[status] = []
            jobs_by_status[status].append(job)

        for status, jobs in jobs_by_status.items():
            status_emoji = {
                "pending": "⏳",
                "running": "🔄",
                "completed": "✅",
                "failed": "❌",
                "cancelled": "🚫"
            }.get(status, "❓")

            result += f"**{status_emoji} {status.title()} ({len(jobs)}):**\n"

            for job in sorted(jobs, key=lambda j: j.created_at, reverse=True)[:5]:  # Show latest 5
                result += f"• {job.url[:50]}...\n"
                result += f"  ID: {job.id}\n"
                result += f"  Created: {format_timestamp(job.created_at)}\n"
                if job.completed_at:
                    result += f"  Completed: {format_timestamp(job.completed_at)}\n"
                result += "\n"

        return result

    async def _export_scraped_data(self, job_id: str, metadata: Dict[str, Any]) -> str:
        """Export scraped data in different formats"""
        try:
            export_format = metadata.get("format", "json")

            if job_id == "all":
                # Export all data
                data_to_export = [asdict(data) for data in self.scraped_data]
                export_title = "All Scraped Data"
            else:
                # Export specific job data
                job_data = [data for data in self.scraped_data if data.job_id == job_id]
                if not job_data:
                    return f"No data found for job ID: {job_id}"
                data_to_export = [asdict(data) for data in job_data]
                export_title = f"Data for Job {job_id}"

            result = f"Data Export: {export_title}\n\n"
            result += f"**Format:** {export_format.upper()}\n"
            result += f"**Records:** {len(data_to_export)}\n\n"

            if export_format == "json":
                result += f"**JSON Export:**\n```json\n{json.dumps(data_to_export, indent=2, default=str)}\n```"
            elif export_format == "csv":
                # Simple CSV representation
                if data_to_export:
                    headers = list(data_to_export[0].keys())
                    result += f"**CSV Export:**\n{','.join(headers)}\n"
                    for record in data_to_export[:10]:  # Limit to 10 records for display
                        values = [str(record.get(h, '')) for h in headers]
                        result += f"{','.join(values)}\n"
                    if len(data_to_export) > 10:
                        result += f"... and {len(data_to_export) - 10} more records\n"
            else:  # text format
                result += f"**Text Export:**\n"
                for i, record in enumerate(data_to_export[:5], 1):
                    result += f"Record {i}:\n"
                    for key, value in record.items():
                        result += f"  {key}: {str(value)[:100]}...\n"
                    result += "\n"

            return result

        except Exception as e:
            self.logger.error(f"Failed to export data: {str(e)}")
            return f"Sorry, I couldn't export the data. Error: {str(e)}"

    # Core scraping methods
    async def _execute_scraping(self, job: ScrapingJob) -> Dict[str, Any]:
        """Execute the actual scraping based on job configuration"""
        try:
            await global_rate_limiter.acquire()  # Respect rate limits

            if job.method == ScrapingMethod.SELENIUM:
                return await self._scrape_with_selenium(job)
            else:  # Default to requests
                return await self._scrape_with_requests(job)

        except Exception as e:
            self.logger.error(f"Scraping execution failed: {str(e)}")
            raise

    async def _scrape_with_requests(self, job: ScrapingJob) -> Dict[str, Any]:
        """Scrape using aiohttp/requests"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(job.url, timeout=30) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}: {response.reason}")

                    html_content = await response.text()

                    if BeautifulSoup is None:
                        # Fallback to basic text extraction
                        return {"raw_html": html_content[:1000]}

                    soup = BeautifulSoup(html_content, 'html.parser')
                    results = {}

                    for data_type in job.target_data:
                        results[data_type] = self._extract_data_by_type(soup, data_type)

                    return results

        except Exception as e:
            self.logger.error(f"Requests scraping failed: {str(e)}")
            raise

    async def _scrape_with_selenium(self, job: ScrapingJob) -> Dict[str, Any]:
        """Scrape using Selenium (for JavaScript-heavy sites)"""
        if webdriver is None:
            raise Exception("Selenium not available. Install selenium package.")

        try:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')

            driver_path = self.selenium_driver_path or "chromedriver"
            driver = webdriver.Chrome(executable_path=driver_path, options=options)

            try:
                driver.get(job.url)

                # Wait for page to load
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )

                results = {}
                for data_type in job.target_data:
                    results[data_type] = self._extract_selenium_data(driver, data_type)

                return results

            finally:
                driver.quit()

        except Exception as e:
            self.logger.error(f"Selenium scraping failed: {str(e)}")
            raise

    def _extract_data_by_type(self, soup: 'BeautifulSoup', data_type: str) -> Any:
        """Extract specific data type from BeautifulSoup object"""
        if data_type not in self.common_selectors:
            # Try to find elements by data_type as class or id
            elements = soup.find_all(class_=data_type) or soup.find_all(id=data_type)
            return [elem.get_text(strip=True) for elem in elements]

        selectors = self.common_selectors[data_type]

        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                if data_type == "links":
                    return [urljoin(soup.base_url if hasattr(soup, 'base_url') else '', elem.get('href', ''))
                           for elem in elements if elem.get('href')]
                elif data_type == "images":
                    return [urljoin(soup.base_url if hasattr(soup, 'base_url') else '', elem.get('src', ''))
                           for elem in elements if elem.get('src')]
                else:
                    return [elem.get_text(strip=True) for elem in elements if elem.get_text(strip=True)]

        return []

    def _extract_selenium_data(self, driver, data_type: str) -> Any:
        """Extract data using Selenium WebDriver"""
        try:
            if data_type == "links":
                elements = driver.find_elements(By.TAG_NAME, "a")
                return [elem.get_attribute("href") for elem in elements if elem.get_attribute("href")]
            elif data_type == "images":
                elements = driver.find_elements(By.TAG_NAME, "img")
                return [elem.get_attribute("src") for elem in elements if elem.get_attribute("src")]
            else:
                # Try common selectors
                selectors = self.common_selectors.get(data_type, [data_type])
                for selector in selectors:
                    try:
                        if selector.startswith('.') or selector.startswith('#'):
                            elements = driver.find_elements(By.CSS_SELECTOR, selector)
                        else:
                            elements = driver.find_elements(By.TAG_NAME, selector)

                        if elements:
                            return [elem.text.strip() for elem in elements if elem.text.strip()]
                    except:
                        continue

                return []

        except Exception as e:
            self.logger.warning(f"Failed to extract {data_type} with Selenium: {str(e)}")
            return []

    async def _scrape_search_engine(self, search_url: str, engine: str, max_results: int) -> List[Dict[str, str]]:
        """Scrape search engine results"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(search_url, timeout=30) as response:
                    html_content = await response.text()

                    if BeautifulSoup is None:
                        return [{"title": "BeautifulSoup not available", "url": search_url, "description": "Install bs4 package"}]

                    soup = BeautifulSoup(html_content, 'html.parser')
                    results = []

                    if engine == "google":
                        # Google search result selectors (may change)
                        result_divs = soup.find_all('div', class_='g')
                        for div in result_divs[:max_results]:
                            title_elem = div.find('h3')
                            link_elem = div.find('a')
                            desc_elem = div.find('span', class_='st') or div.find('div', class_='s')

                            if title_elem and link_elem:
                                results.append({
                                    'title': title_elem.get_text(strip=True),
                                    'url': link_elem.get('href', ''),
                                    'description': desc_elem.get_text(strip=True) if desc_elem else ''
                                })

                    # Add fallback for other engines or if no results found
                    if not results:
                        results.append({
                            'title': f'Search results for {engine}',
                            'url': search_url,
                            'description': 'Search engine scraping may be limited due to anti-bot measures'
                        })

                    return results

        except Exception as e:
            self.logger.error(f"Search engine scraping failed: {str(e)}")
            return [{"title": "Error", "url": search_url, "description": str(e)}]

    # Database methods
    def _save_scraping_job(self, job: ScrapingJob):
        """Save scraping job to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO scraping_jobs
                (id, url, target_data, method, status, created_at, completed_at, results, error_message, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                job.id, job.url, json.dumps(job.target_data), job.method.value,
                job.status.value, job.created_at.isoformat(),
                job.completed_at.isoformat() if job.completed_at else None,
                json.dumps(job.results) if job.results else None,
                job.error_message, json.dumps(job.metadata)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Failed to save scraping job: {str(e)}")

    def _save_scraped_data(self, data: ScrapedData):
        """Save scraped data to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO scraped_data
                (id, job_id, url, data_type, content, scraped_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                data.id, data.job_id, data.url, data.data_type,
                json.dumps(data.content), data.scraped_at.isoformat()
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Failed to save scraped data: {str(e)}")
