"""
Email Generation Agent - Handles automated email composition and management
"""
import smtplib
import imaplib
import email
import json
import re
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import sqlite3

from .base_agent import BaseAgent, AgentTask
from .utils import sanitize_input, generate_task_id, format_timestamp, validate_email
from config import config

class EmailType(Enum):
    """Types of emails"""
    PROFESSIONAL = "professional"
    PERSONAL = "personal"
    MARKETING = "marketing"
    FOLLOW_UP = "follow_up"
    THANK_YOU = "thank_you"
    INVITATION = "invitation"
    ANNOUNCEMENT = "announcement"
    COMPLAINT = "complaint"
    INQUIRY = "inquiry"
    NEWSLETTER = "newsletter"

class EmailTone(Enum):
    """Email tone options"""
    FORMAL = "formal"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    URGENT = "urgent"
    APOLOGETIC = "apologetic"
    ENTHUSIASTIC = "enthusiastic"

class EmailPriority(Enum):
    """Email priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

@dataclass
class EmailTemplate:
    """Represents an email template"""
    id: str
    name: str
    email_type: EmailType
    subject_template: str
    body_template: str
    tone: EmailTone
    variables: List[str] = None  # Variables that can be replaced
    tags: List[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.variables is None:
            self.variables = []
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class EmailDraft:
    """Represents an email draft"""
    id: str
    to_addresses: List[str]
    cc_addresses: List[str] = None
    bcc_addresses: List[str] = None
    subject: str = ""
    body: str = ""
    email_type: EmailType = EmailType.PROFESSIONAL
    tone: EmailTone = EmailTone.PROFESSIONAL
    priority: EmailPriority = EmailPriority.NORMAL
    attachments: List[str] = None
    is_html: bool = False
    created_at: datetime = None
    sent_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.cc_addresses is None:
            self.cc_addresses = []
        if self.bcc_addresses is None:
            self.bcc_addresses = []
        if self.attachments is None:
            self.attachments = []
        if self.created_at is None:
            self.created_at = datetime.now()

class EmailAgent(BaseAgent):
    """Email Generation Agent for email composition and management"""
    
    def __init__(self):
        super().__init__(
            name="email_agent",
            description="Handles automated email composition and management"
        )
        
        self.email_templates: List[EmailTemplate] = []
        self.email_drafts: List[EmailDraft] = []
        self.db_path = "email_agent.db"
        
        # Initialize database
        self._init_database()
        
        # Load existing data
        self._load_data()
        
        # Email configuration
        self.smtp_server = config.smtp_server
        self.smtp_port = config.smtp_port
        self.email_username = config.email_username
        self.email_password = config.email_password
        
        # Initialize default templates
        self._create_default_templates()
    
    def _init_database(self):
        """Initialize SQLite database for email data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create email templates table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS email_templates (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email_type TEXT NOT NULL,
                    subject_template TEXT NOT NULL,
                    body_template TEXT NOT NULL,
                    tone TEXT NOT NULL,
                    variables TEXT,
                    tags TEXT,
                    created_at TEXT NOT NULL
                )
            ''')
            
            # Create email drafts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS email_drafts (
                    id TEXT PRIMARY KEY,
                    to_addresses TEXT NOT NULL,
                    cc_addresses TEXT,
                    bcc_addresses TEXT,
                    subject TEXT,
                    body TEXT,
                    email_type TEXT NOT NULL,
                    tone TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    attachments TEXT,
                    is_html BOOLEAN DEFAULT FALSE,
                    created_at TEXT NOT NULL,
                    sent_at TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Email database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize email database: {str(e)}")
    
    def _load_data(self):
        """Load email templates and drafts from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load email templates
            cursor.execute("SELECT * FROM email_templates")
            for row in cursor.fetchall():
                template = EmailTemplate(
                    id=row[0],
                    name=row[1],
                    email_type=EmailType(row[2]),
                    subject_template=row[3],
                    body_template=row[4],
                    tone=EmailTone(row[5]),
                    variables=json.loads(row[6]) if row[6] else [],
                    tags=json.loads(row[7]) if row[7] else [],
                    created_at=datetime.fromisoformat(row[8])
                )
                self.email_templates.append(template)
            
            # Load email drafts
            cursor.execute("SELECT * FROM email_drafts")
            for row in cursor.fetchall():
                draft = EmailDraft(
                    id=row[0],
                    to_addresses=json.loads(row[1]),
                    cc_addresses=json.loads(row[2]) if row[2] else [],
                    bcc_addresses=json.loads(row[3]) if row[3] else [],
                    subject=row[4],
                    body=row[5],
                    email_type=EmailType(row[6]),
                    tone=EmailTone(row[7]),
                    priority=EmailPriority(row[8]),
                    attachments=json.loads(row[9]) if row[9] else [],
                    is_html=bool(row[10]),
                    created_at=datetime.fromisoformat(row[11]),
                    sent_at=datetime.fromisoformat(row[12]) if row[12] else None
                )
                self.email_drafts.append(draft)
            
            conn.close()
            self.logger.info(f"Loaded {len(self.email_templates)} templates and {len(self.email_drafts)} drafts")
            
        except Exception as e:
            self.logger.error(f"Failed to load email data: {str(e)}")
    
    async def _process_task_impl(self, task: AgentTask) -> str:
        """Process email agent tasks"""
        task_type = task.task_type.lower()
        content = sanitize_input(task.content)
        
        if task_type == "compose_email":
            return await self._compose_email(content, task.metadata)
        elif task_type == "generate_subject":
            return await self._generate_subject(content, task.metadata)
        elif task_type == "improve_email":
            return await self._improve_email(content, task.metadata)
        elif task_type == "create_template":
            return await self._create_template(content, task.metadata)
        elif task_type == "use_template":
            return await self._use_template(content, task.metadata)
        elif task_type == "send_email":
            return await self._send_email(content, task.metadata)
        elif task_type == "list_drafts":
            return await self._list_drafts()
        elif task_type == "list_templates":
            return await self._list_templates()
        elif task_type == "format_email":
            return await self._format_email(content, task.metadata)
        elif task_type == "health_check":
            return "Email Agent is ready to help with all your email composition needs!"
        else:
            return await self._compose_email(content, task.metadata)
    
    async def _compose_email(self, description: str, metadata: Dict[str, Any]) -> str:
        """Compose an email based on description"""
        try:
            # Extract email parameters
            to_addresses = metadata.get("to", [])
            if isinstance(to_addresses, str):
                to_addresses = [to_addresses]
            
            email_type = metadata.get("email_type", "professional")
            tone = metadata.get("tone", "professional")
            subject_hint = metadata.get("subject", "")
            
            try:
                email_type_enum = EmailType(email_type)
                tone_enum = EmailTone(tone)
            except ValueError:
                email_type_enum = EmailType.PROFESSIONAL
                tone_enum = EmailTone.PROFESSIONAL
            
            # Generate email content
            email_content = await self._generate_email_content(
                description, email_type_enum, tone_enum, subject_hint
            )
            
            # Create draft
            draft = EmailDraft(
                id=generate_task_id(),
                to_addresses=to_addresses,
                cc_addresses=metadata.get("cc", []),
                bcc_addresses=metadata.get("bcc", []),
                subject=email_content["subject"],
                body=email_content["body"],
                email_type=email_type_enum,
                tone=tone_enum,
                priority=EmailPriority(metadata.get("priority", "normal"))
            )
            
            # Save draft
            self._save_email_draft(draft)
            self.email_drafts.append(draft)
            
            result = f"Email Composed\n\n"
            result += f"**Draft ID:** {draft.id}\n"
            result += f"**Type:** {email_type_enum.value.title()}\n"
            result += f"**Tone:** {tone_enum.value.title()}\n"
            if to_addresses:
                result += f"**To:** {', '.join(to_addresses)}\n"
            result += f"\n**Subject:** {draft.subject}\n\n"
            result += f"**Body:**\n{draft.body}\n\n"
            
            # Add suggestions
            suggestions = self._get_email_suggestions(draft)
            if suggestions:
                result += f"**Suggestions:**\n"
                for suggestion in suggestions:
                    result += f"• {suggestion}\n"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to compose email: {str(e)}")
            return f"Sorry, I couldn't compose the email. Error: {str(e)}"
    
    async def _generate_subject(self, email_content: str, metadata: Dict[str, Any]) -> str:
        """Generate email subject lines"""
        try:
            email_type = metadata.get("email_type", "professional")
            tone = metadata.get("tone", "professional")
            count = metadata.get("count", 5)
            
            # Generate multiple subject line options
            prompt = f"""
            Generate {count} compelling email subject lines for the following email content:
            
            Email Content: {email_content}
            Email Type: {email_type}
            Tone: {tone}
            
            Make the subject lines:
            1. Clear and specific
            2. Appropriate for the tone and type
            3. Engaging but not spammy
            4. Under 50 characters when possible
            
            Provide variety in style and approach.
            """
            
            subject_suggestions = self.generate_response(prompt, max_length=512)
            
            result = f"Subject Line Suggestions\n\n"
            result += f"**Email Type:** {email_type.title()}\n"
            result += f"**Tone:** {tone.title()}\n\n"
            result += subject_suggestions
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate subject: {str(e)}")
            return f"Sorry, I couldn't generate subject lines. Error: {str(e)}"
    
    async def _improve_email(self, email_content: str, metadata: Dict[str, Any]) -> str:
        """Improve an existing email"""
        try:
            improvement_focus = metadata.get("focus", "overall")  # clarity, tone, persuasion, brevity
            target_tone = metadata.get("target_tone", "professional")
            
            # Analyze and improve the email
            improvement_prompt = f"""
            Improve the following email with focus on: {improvement_focus}
            Target tone: {target_tone}
            
            Original Email:
            {email_content}
            
            Provide:
            1. Improved version of the email
            2. Explanation of changes made
            3. Additional suggestions
            
            Make it more effective while maintaining the original intent.
            """
            
            improved_content = self.generate_response(improvement_prompt, max_length=1024)
            
            result = f"Email Improvement\n\n"
            result += f"**Focus:** {improvement_focus.title()}\n"
            result += f"**Target Tone:** {target_tone.title()}\n\n"
            result += improved_content
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to improve email: {str(e)}")
            return f"Sorry, I couldn't improve the email. Error: {str(e)}"
    
    async def _create_template(self, template_description: str, metadata: Dict[str, Any]) -> str:
        """Create a new email template"""
        try:
            name = metadata.get("name", template_description[:50])
            email_type = metadata.get("email_type", "professional")
            tone = metadata.get("tone", "professional")
            
            # Generate template content
            template_content = await self._generate_template_content(
                template_description, email_type, tone
            )
            
            template = EmailTemplate(
                id=generate_task_id(),
                name=name,
                email_type=EmailType(email_type),
                subject_template=template_content["subject"],
                body_template=template_content["body"],
                tone=EmailTone(tone),
                variables=template_content.get("variables", []),
                tags=metadata.get("tags", [])
            )
            
            # Save template
            self._save_email_template(template)
            self.email_templates.append(template)
            
            result = f"Email Template Created\n\n"
            result += f"**Template ID:** {template.id}\n"
            result += f"**Name:** {template.name}\n"
            result += f"**Type:** {template.email_type.value.title()}\n"
            result += f"**Tone:** {template.tone.value.title()}\n\n"
            
            if template.variables:
                result += f"**Variables:** {', '.join(template.variables)}\n\n"
            
            result += f"**Subject Template:**\n{template.subject_template}\n\n"
            result += f"**Body Template:**\n{template.body_template}\n"
            
            return result

        except Exception as e:
            self.logger.error(f"Failed to create template: {str(e)}")
            return f"Sorry, I couldn't create the template. Error: {str(e)}"

    async def _use_template(self, template_identifier: str, metadata: Dict[str, Any]) -> str:
        """Use an existing template to create an email"""
        try:
            # Find template
            template = None
            for t in self.email_templates:
                if t.id == template_identifier or template_identifier.lower() in t.name.lower():
                    template = t
                    break

            if not template:
                return f"Template '{template_identifier}' not found."

            # Get variable values
            variables = metadata.get("variables", {})

            # Replace variables in template
            subject = template.subject_template
            body = template.body_template

            for var in template.variables:
                placeholder = f"{{{var}}}"
                if var in variables:
                    subject = subject.replace(placeholder, str(variables[var]))
                    body = body.replace(placeholder, str(variables[var]))

            # Create draft
            draft = EmailDraft(
                id=generate_task_id(),
                to_addresses=metadata.get("to", []),
                cc_addresses=metadata.get("cc", []),
                bcc_addresses=metadata.get("bcc", []),
                subject=subject,
                body=body,
                email_type=template.email_type,
                tone=template.tone,
                priority=EmailPriority(metadata.get("priority", "normal"))
            )

            # Save draft
            self._save_email_draft(draft)
            self.email_drafts.append(draft)

            result = f"Email Created from Template\n\n"
            result += f"**Template:** {template.name}\n"
            result += f"**Draft ID:** {draft.id}\n\n"
            result += f"**Subject:** {draft.subject}\n\n"
            result += f"**Body:**\n{draft.body}\n"

            # Show unfilled variables
            unfilled_vars = [var for var in template.variables if f"{{{var}}}" in subject or f"{{{var}}}" in body]
            if unfilled_vars:
                result += f"\n**Note:** The following variables still need values: {', '.join(unfilled_vars)}"

            return result

        except Exception as e:
            self.logger.error(f"Failed to use template: {str(e)}")
            return f"Sorry, I couldn't use the template. Error: {str(e)}"

    async def _send_email(self, draft_identifier: str, metadata: Dict[str, Any]) -> str:
        """Send an email draft"""
        try:
            # Find draft
            draft = None
            for d in self.email_drafts:
                if d.id == draft_identifier:
                    draft = d
                    break

            if not draft:
                return f"Draft '{draft_identifier}' not found."

            if not self.smtp_server or not self.email_username or not self.email_password:
                return "Email configuration is not set up. Please configure SMTP settings."

            # Validate email addresses
            invalid_emails = []
            for email_addr in draft.to_addresses + draft.cc_addresses + draft.bcc_addresses:
                if not validate_email(email_addr):
                    invalid_emails.append(email_addr)

            if invalid_emails:
                return f"Invalid email addresses: {', '.join(invalid_emails)}"

            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.email_username
            msg['To'] = ', '.join(draft.to_addresses)
            if draft.cc_addresses:
                msg['Cc'] = ', '.join(draft.cc_addresses)
            msg['Subject'] = draft.subject

            # Add priority header
            if draft.priority != EmailPriority.NORMAL:
                priority_map = {
                    EmailPriority.LOW: "5",
                    EmailPriority.HIGH: "2",
                    EmailPriority.URGENT: "1"
                }
                msg['X-Priority'] = priority_map.get(draft.priority, "3")

            # Add body
            if draft.is_html:
                msg.attach(MIMEText(draft.body, 'html'))
            else:
                msg.attach(MIMEText(draft.body, 'plain'))

            # Add attachments (if any)
            for attachment_path in draft.attachments:
                try:
                    with open(attachment_path, "rb") as attachment:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(attachment.read())

                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {attachment_path.split("/")[-1]}'
                    )
                    msg.attach(part)
                except Exception as e:
                    self.logger.warning(f"Failed to attach file {attachment_path}: {str(e)}")

            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_username, self.email_password)

            all_recipients = draft.to_addresses + draft.cc_addresses + draft.bcc_addresses
            server.sendmail(self.email_username, all_recipients, msg.as_string())
            server.quit()

            # Update draft
            draft.sent_at = datetime.now()
            self._save_email_draft(draft)

            result = f"Email Sent Successfully\n\n"
            result += f"**To:** {', '.join(draft.to_addresses)}\n"
            if draft.cc_addresses:
                result += f"**CC:** {', '.join(draft.cc_addresses)}\n"
            result += f"**Subject:** {draft.subject}\n"
            result += f"**Sent At:** {format_timestamp(draft.sent_at)}\n"

            return result

        except Exception as e:
            self.logger.error(f"Failed to send email: {str(e)}")
            return f"Sorry, I couldn't send the email. Error: {str(e)}"

    async def _list_drafts(self) -> str:
        """List all email drafts"""
        if not self.email_drafts:
            return "No email drafts found."

        result = "Email Drafts\n\n"

        # Separate sent and unsent drafts
        unsent_drafts = [d for d in self.email_drafts if d.sent_at is None]
        sent_drafts = [d for d in self.email_drafts if d.sent_at is not None]

        if unsent_drafts:
            result += f"**Unsent Drafts ({len(unsent_drafts)}):**\n"
            for draft in sorted(unsent_drafts, key=lambda d: d.created_at, reverse=True):
                result += f"📝 **{draft.subject or 'No Subject'}**\n"
                result += f"   ID: {draft.id}\n"
                result += f"   To: {', '.join(draft.to_addresses) if draft.to_addresses else 'Not specified'}\n"
                result += f"   Type: {draft.email_type.value.title()}\n"
                result += f"   Created: {format_timestamp(draft.created_at)}\n\n"

        if sent_drafts:
            result += f"**Sent Emails ({len(sent_drafts)}):**\n"
            for draft in sorted(sent_drafts, key=lambda d: d.sent_at, reverse=True)[:10]:  # Show last 10
                result += f"✅ **{draft.subject or 'No Subject'}**\n"
                result += f"   ID: {draft.id}\n"
                result += f"   To: {', '.join(draft.to_addresses)}\n"
                result += f"   Sent: {format_timestamp(draft.sent_at)}\n\n"

        return result

    async def _list_templates(self) -> str:
        """List all email templates"""
        if not self.email_templates:
            return "No email templates found."

        result = "Email Templates\n\n"

        # Group by type
        templates_by_type = {}
        for template in self.email_templates:
            email_type = template.email_type.value
            if email_type not in templates_by_type:
                templates_by_type[email_type] = []
            templates_by_type[email_type].append(template)

        for email_type, templates in templates_by_type.items():
            result += f"**{email_type.title()} Templates:**\n"
            for template in templates:
                result += f"📄 **{template.name}**\n"
                result += f"   ID: {template.id}\n"
                result += f"   Tone: {template.tone.value.title()}\n"
                if template.variables:
                    result += f"   Variables: {', '.join(template.variables)}\n"
                result += f"   Created: {format_timestamp(template.created_at)}\n\n"

        return result

    async def _format_email(self, email_content: str, metadata: Dict[str, Any]) -> str:
        """Format an email for better presentation"""
        try:
            format_type = metadata.get("format", "professional")  # professional, html, plain

            if format_type == "html":
                formatted_content = self._convert_to_html(email_content)
            elif format_type == "plain":
                formatted_content = self._convert_to_plain_text(email_content)
            else:  # professional
                formatted_content = self._format_professional(email_content)

            result = f"Formatted Email ({format_type.title()})\n\n"
            result += formatted_content

            return result

        except Exception as e:
            self.logger.error(f"Failed to format email: {str(e)}")
            return f"Sorry, I couldn't format the email. Error: {str(e)}"

    # Helper methods
    async def _generate_email_content(self, description: str, email_type: EmailType,
                                    tone: EmailTone, subject_hint: str) -> Dict[str, str]:
        """Generate email content using LLM"""
        prompt = f"""
        Compose a {tone.value} {email_type.value} email based on the following description:

        Description: {description}
        Subject hint: {subject_hint}

        Generate:
        1. An appropriate subject line
        2. A well-structured email body

        Make sure the email:
        - Has the right tone ({tone.value})
        - Is appropriate for the type ({email_type.value})
        - Is clear and actionable
        - Follows proper email etiquette
        """

        content = self.generate_response(prompt, max_length=1024)

        # Parse the generated content to extract subject and body
        lines = content.split('\n')
        subject = ""
        body_lines = []

        for i, line in enumerate(lines):
            if line.lower().startswith('subject:'):
                subject = line.split(':', 1)[1].strip()
            elif subject and line.strip():  # After subject is found
                body_lines.extend(lines[i:])
                break

        if not subject and subject_hint:
            subject = subject_hint
        elif not subject:
            subject = f"Re: {description[:30]}..."

        body = '\n'.join(body_lines).strip() if body_lines else content

        return {
            "subject": subject,
            "body": body
        }

    async def _generate_template_content(self, description: str, email_type: str, tone: str) -> Dict[str, Any]:
        """Generate template content with variables"""
        prompt = f"""
        Create an email template for: {description}
        Type: {email_type}
        Tone: {tone}

        Create a template with:
        1. Subject line with variables in {{variable_name}} format
        2. Body with variables for personalization
        3. List the variables used

        Make it reusable for similar situations.
        """

        content = self.generate_response(prompt, max_length=1024)

        # Extract variables from the content
        variables = re.findall(r'\{([^}]+)\}', content)
        variables = list(set(variables))  # Remove duplicates

        # Parse subject and body (simplified)
        lines = content.split('\n')
        subject = ""
        body_lines = []

        for i, line in enumerate(lines):
            if line.lower().startswith('subject:'):
                subject = line.split(':', 1)[1].strip()
            elif subject and line.strip():
                body_lines.extend(lines[i:])
                break

        body = '\n'.join(body_lines).strip() if body_lines else content

        return {
            "subject": subject or f"{email_type.title()} Email Template",
            "body": body,
            "variables": variables
        }

    def _get_email_suggestions(self, draft: EmailDraft) -> List[str]:
        """Get suggestions for improving the email"""
        suggestions = []

        # Check subject line
        if not draft.subject:
            suggestions.append("Add a clear and specific subject line")
        elif len(draft.subject) > 50:
            suggestions.append("Consider shortening the subject line (currently over 50 characters)")

        # Check body length
        if len(draft.body) < 50:
            suggestions.append("The email body seems quite short - consider adding more context")
        elif len(draft.body) > 2000:
            suggestions.append("The email is quite long - consider breaking it into shorter paragraphs")

        # Check for call to action
        action_words = ['please', 'kindly', 'could you', 'would you', 'let me know', 'respond', 'reply']
        if not any(word in draft.body.lower() for word in action_words):
            suggestions.append("Consider adding a clear call to action")

        # Check greeting and closing
        if not any(greeting in draft.body.lower() for greeting in ['dear', 'hello', 'hi', 'greetings']):
            suggestions.append("Consider adding a greeting")

        if not any(closing in draft.body.lower() for closing in ['regards', 'sincerely', 'best', 'thanks']):
            suggestions.append("Consider adding a professional closing")

        return suggestions

    def _convert_to_html(self, content: str) -> str:
        """Convert plain text to HTML format"""
        html_content = content.replace('\n\n', '</p><p>')
        html_content = html_content.replace('\n', '<br>')
        return f"<html><body><p>{html_content}</p></body></html>"

    def _convert_to_plain_text(self, content: str) -> str:
        """Convert to plain text (remove any HTML)"""
        # Simple HTML tag removal
        clean_text = re.sub(r'<[^>]+>', '', content)
        return clean_text.strip()

    def _format_professional(self, content: str) -> str:
        """Format email in professional style"""
        # Add proper spacing and structure
        lines = content.split('\n')
        formatted_lines = []

        for line in lines:
            if line.strip():
                formatted_lines.append(line.strip())
            else:
                formatted_lines.append('')

        return '\n'.join(formatted_lines)

    def _save_email_template(self, template: EmailTemplate):
        """Save email template to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO email_templates
                (id, name, email_type, subject_template, body_template, tone, variables, tags, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                template.id, template.name, template.email_type.value,
                template.subject_template, template.body_template, template.tone.value,
                json.dumps(template.variables), json.dumps(template.tags),
                template.created_at.isoformat()
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Failed to save email template: {str(e)}")

    def _save_email_draft(self, draft: EmailDraft):
        """Save email draft to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT OR REPLACE INTO email_drafts
                (id, to_addresses, cc_addresses, bcc_addresses, subject, body, email_type, tone, priority, attachments, is_html, created_at, sent_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                draft.id, json.dumps(draft.to_addresses), json.dumps(draft.cc_addresses),
                json.dumps(draft.bcc_addresses), draft.subject, draft.body,
                draft.email_type.value, draft.tone.value, draft.priority.value,
                json.dumps(draft.attachments), draft.is_html,
                draft.created_at.isoformat(),
                draft.sent_at.isoformat() if draft.sent_at else None
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            self.logger.error(f"Failed to save email draft: {str(e)}")

    def _create_default_templates(self):
        """Create default email templates if none exist"""
        if self.email_templates:
            return  # Templates already exist

        default_templates = [
            {
                "name": "Professional Follow-up",
                "email_type": EmailType.FOLLOW_UP,
                "tone": EmailTone.PROFESSIONAL,
                "subject": "Following up on {topic}",
                "body": "Dear {name},\n\nI hope this email finds you well. I wanted to follow up on our previous discussion regarding {topic}.\n\n{main_content}\n\nI would appreciate your thoughts on this matter. Please let me know if you need any additional information.\n\nBest regards,\n{sender_name}",
                "variables": ["name", "topic", "main_content", "sender_name"]
            },
            {
                "name": "Thank You Note",
                "email_type": EmailType.THANK_YOU,
                "tone": EmailTone.FRIENDLY,
                "subject": "Thank you for {reason}",
                "body": "Hi {name},\n\nI wanted to take a moment to thank you for {reason}. {specific_details}\n\nYour {contribution} made a real difference, and I truly appreciate it.\n\nWarm regards,\n{sender_name}",
                "variables": ["name", "reason", "specific_details", "contribution", "sender_name"]
            }
        ]

        for template_data in default_templates:
            template = EmailTemplate(
                id=generate_task_id(),
                name=template_data["name"],
                email_type=template_data["email_type"],
                subject_template=template_data["subject"],
                body_template=template_data["body"],
                tone=template_data["tone"],
                variables=template_data["variables"]
            )

            self._save_email_template(template)
            self.email_templates.append(template)
