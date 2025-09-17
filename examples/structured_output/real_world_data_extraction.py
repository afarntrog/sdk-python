#!/usr/bin/env python3
"""
Real-World Data Extraction Example

This example demonstrates practical use cases for structured output in real-world scenarios:
1. Extracting contact information from unstructured text
2. Parsing invoice data
3. Analyzing customer feedback
4. Processing meeting notes

This shows the power of structured output for data extraction and processing tasks.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Union
from pydantic import BaseModel, Field, validator
from strands import Agent, ToolOutput


# === Contact Information Extraction ===

class ContactInfo(BaseModel):
    """Structured contact information."""

    name: str = Field(description="Full name of the person")
    email: Optional[str] = Field(default=None, description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    company: Optional[str] = Field(default=None, description="Company name")
    title: Optional[str] = Field(default=None, description="Job title")
    address: Optional[str] = Field(default=None, description="Physical address")

    @validator('email')
    def validate_email(cls, v):
        if v and '@' not in v:
            raise ValueError('Invalid email format')
        return v


# === Invoice Processing ===

class InvoiceLineItem(BaseModel):
    """A single line item on an invoice."""

    description: str = Field(description="Description of the item or service")
    quantity: int = Field(description="Quantity ordered")
    unit_price: float = Field(description="Price per unit")
    total_price: float = Field(description="Total price for this line item")


class Invoice(BaseModel):
    """Structured invoice data."""

    invoice_number: str = Field(description="Invoice number")
    date: str = Field(description="Invoice date")
    due_date: Optional[str] = Field(default=None, description="Due date for payment")
    vendor_name: str = Field(description="Name of the vendor/company")
    vendor_address: Optional[str] = Field(default=None, description="Vendor address")
    bill_to_name: str = Field(description="Name of the customer being billed")
    bill_to_address: Optional[str] = Field(default=None, description="Customer address")
    line_items: List[InvoiceLineItem] = Field(description="List of invoice line items")
    subtotal: float = Field(description="Subtotal before taxes")
    tax_amount: Optional[float] = Field(default=None, description="Tax amount")
    total_amount: float = Field(description="Total amount due")


# === Customer Feedback Analysis ===

class SentimentScore(str, Enum):
    """Sentiment classification."""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


class FeedbackCategory(str, Enum):
    """Categories for customer feedback."""
    PRODUCT_QUALITY = "product_quality"
    CUSTOMER_SERVICE = "customer_service"
    PRICING = "pricing"
    SHIPPING = "shipping"
    WEBSITE_UX = "website_ux"
    BILLING = "billing"
    OTHER = "other"


class CustomerFeedback(BaseModel):
    """Structured customer feedback analysis."""

    customer_id: Optional[str] = Field(default=None, description="Customer identifier if available")
    sentiment: SentimentScore = Field(description="Overall sentiment of the feedback")
    categories: List[FeedbackCategory] = Field(description="Categories this feedback relates to")
    key_issues: List[str] = Field(description="Main issues or complaints mentioned")
    positive_mentions: List[str] = Field(description="Positive aspects mentioned")
    urgency_level: int = Field(description="Urgency level from 1 (low) to 5 (high)", ge=1, le=5)
    action_required: bool = Field(description="Whether immediate action is required")
    summary: str = Field(description="Brief summary of the feedback")


# === Meeting Notes Processing ===

class ActionItem(BaseModel):
    """An action item from a meeting."""

    task: str = Field(description="Description of the task")
    assignee: Optional[str] = Field(default=None, description="Person assigned to the task")
    due_date: Optional[str] = Field(default=None, description="Due date for completion")
    priority: str = Field(description="Priority level (low, medium, high)")


class Decision(BaseModel):
    """A decision made during the meeting."""

    decision: str = Field(description="What was decided")
    rationale: Optional[str] = Field(default=None, description="Reasoning behind the decision")


class MeetingNotes(BaseModel):
    """Structured meeting notes."""

    meeting_title: str = Field(description="Title or purpose of the meeting")
    date: str = Field(description="Meeting date")
    attendees: List[str] = Field(description="List of attendees")
    key_topics: List[str] = Field(description="Main topics discussed")
    decisions: List[Decision] = Field(description="Decisions made")
    action_items: List[ActionItem] = Field(description="Action items assigned")
    next_meeting: Optional[str] = Field(default=None, description="Next meeting date if scheduled")


def extract_contact_info():
    """Demonstrate contact information extraction from unstructured text."""
    print("📇 Contact Information Extraction")
    print("=" * 50)

    agent = Agent(model_id="gpt-4o", output_mode=ToolOutput())

    # Sample unstructured text with contact information
    contact_text = """
    Hi there! My name is Sarah Johnson and I'm the Marketing Director at TechCorp Solutions.
    You can reach me at sarah.johnson@techcorp.com or call me at (555) 123-4567.
    Our office is located at 123 Business Ave, Suite 456, San Francisco, CA 94105.
    I'd love to discuss potential collaboration opportunities!
    """

    print("📄 Input text:")
    print(contact_text.strip())

    result = agent(
        f"Extract contact information from this text: {contact_text}",
        output_type=ContactInfo
    )

    contact = result.get_structured_output(ContactInfo)

    print("\n📊 Extracted Contact Information:")
    print(f"👤 Name: {contact.name}")
    print(f"📧 Email: {contact.email}")
    print(f"📞 Phone: {contact.phone}")
    print(f"🏢 Company: {contact.company}")
    print(f"💼 Title: {contact.title}")
    print(f"📍 Address: {contact.address}")


def process_invoice():
    """Demonstrate invoice data extraction."""
    print("\n\n📄 Invoice Processing")
    print("=" * 50)

    agent = Agent(model_id="gpt-4o", output_mode=ToolOutput())

    # Sample invoice text
    invoice_text = """
    INVOICE #INV-2024-001

    Date: March 15, 2024
    Due Date: April 15, 2024

    From:
    WebDev Services LLC
    456 Tech Street
    Austin, TX 78701

    Bill To:
    ABC Corporation
    789 Business Blvd
    Houston, TX 77001

    Description                 Qty    Unit Price    Total
    Website Development          1      $5,000.00    $5,000.00
    Logo Design                  1      $500.00      $500.00
    Hosting Setup               1      $200.00      $200.00

    Subtotal:                                       $5,700.00
    Tax (8.25%):                                    $470.25
    Total:                                          $6,170.25
    """

    print("📄 Input invoice:")
    print(invoice_text.strip())

    result = agent(
        f"Extract structured data from this invoice: {invoice_text}",
        output_type=Invoice
    )

    invoice = result.get_structured_output(Invoice)

    print("\n📊 Extracted Invoice Data:")
    print(f"🔢 Invoice #: {invoice.invoice_number}")
    print(f"📅 Date: {invoice.date}")
    print(f"🏢 Vendor: {invoice.vendor_name}")
    print(f"👤 Bill To: {invoice.bill_to_name}")
    print(f"💰 Total: ${invoice.total_amount:,.2f}")

    print("\n📝 Line Items:")
    for i, item in enumerate(invoice.line_items, 1):
        print(f"  {i}. {item.description}: {item.quantity} × ${item.unit_price:,.2f} = ${item.total_price:,.2f}")


def analyze_customer_feedback():
    """Demonstrate customer feedback analysis."""
    print("\n\n💬 Customer Feedback Analysis")
    print("=" * 50)

    agent = Agent(model_id="gpt-4o", output_mode=ToolOutput())

    # Sample customer feedback
    feedback_text = """
    I'm really disappointed with my recent order #12345. The product quality was much worse
    than expected - the material feels cheap and flimsy. It took 3 weeks to arrive when I
    was promised 5-7 business days. When I tried to contact customer service about the delay,
    I was on hold for over an hour and then got disconnected! This is unacceptable for a
    $200 purchase. I need a refund or replacement immediately.

    However, I will say that your website is easy to navigate and the checkout process was smooth.
    """

    print("📄 Customer feedback:")
    print(feedback_text.strip())

    result = agent(
        f"Analyze this customer feedback and extract structured data: {feedback_text}",
        output_type=CustomerFeedback
    )

    feedback = result.get_structured_output(CustomerFeedback)

    print("\n📊 Feedback Analysis:")
    print(f"😊 Sentiment: {feedback.sentiment.value.replace('_', ' ').title()}")
    print(f"🏷️  Categories: {', '.join([cat.value.replace('_', ' ').title() for cat in feedback.categories])}")
    print(f"⚠️  Urgency Level: {feedback.urgency_level}/5")
    print(f"🚨 Action Required: {'Yes' if feedback.action_required else 'No'}")

    print("\n❌ Key Issues:")
    for issue in feedback.key_issues:
        print(f"  • {issue}")

    print("\n✅ Positive Mentions:")
    for positive in feedback.positive_mentions:
        print(f"  • {positive}")

    print(f"\n📝 Summary: {feedback.summary}")


def process_meeting_notes():
    """Demonstrate meeting notes processing."""
    print("\n\n📋 Meeting Notes Processing")
    print("=" * 50)

    agent = Agent(model_id="gpt-4o", output_mode=ToolOutput())

    # Sample meeting notes
    notes_text = """
    Weekly Team Standup - March 18, 2024

    Attendees: John Smith (Manager), Sarah Davis (Developer), Mike Wilson (Designer), Lisa Chen (QA)

    Topics Discussed:
    - Sprint progress review
    - Bug reports from last release
    - New feature requirements for Q2
    - Team vacation schedule

    Key Decisions:
    1. Decided to extend current sprint by 3 days to fix critical bugs
    2. Approved budget for new testing tools ($5,000)
    3. Lisa will lead the QA process improvement initiative

    Action Items:
    - Sarah: Fix login bug by Friday (high priority)
    - Mike: Complete new UI mockups by March 22
    - John: Schedule client meeting for feature review (medium priority)
    - Lisa: Research testing tools and provide recommendations by March 25

    Next meeting: March 25, 2024
    """

    print("📄 Meeting notes:")
    print(notes_text.strip())

    result = agent(
        f"Extract structured information from these meeting notes: {notes_text}",
        output_type=MeetingNotes
    )

    notes = result.get_structured_output(MeetingNotes)

    print("\n📊 Structured Meeting Data:")
    print(f"📅 Meeting: {notes.meeting_title} ({notes.date})")
    print(f"👥 Attendees: {', '.join(notes.attendees)}")

    print(f"\n🎯 Key Topics:")
    for topic in notes.key_topics:
        print(f"  • {topic}")

    print(f"\n✅ Decisions Made:")
    for i, decision in enumerate(notes.decisions, 1):
        print(f"  {i}. {decision.decision}")
        if decision.rationale:
            print(f"     Rationale: {decision.rationale}")

    print(f"\n📋 Action Items:")
    for item in notes.action_items:
        assignee = f" ({item.assignee})" if item.assignee else ""
        due_date = f" - Due: {item.due_date}" if item.due_date else ""
        print(f"  • [{item.priority.upper()}] {item.task}{assignee}{due_date}")

    if notes.next_meeting:
        print(f"\n📅 Next Meeting: {notes.next_meeting}")


def main():
    """Run all data extraction examples."""
    print("🎯 Real-World Data Extraction Examples")
    print("=" * 70)

    # Run all examples
    extract_contact_info()
    process_invoice()
    analyze_customer_feedback()
    process_meeting_notes()

    print("\n" + "=" * 70)
    print("✅ All Data Extraction Examples Complete!")

    print("\n🎓 Key Benefits Demonstrated:")
    print("• Convert unstructured text into structured, searchable data")
    print("• Validate and clean data automatically with Pydantic")
    print("• Enable automated processing and analysis workflows")
    print("• Standardize data formats across different input types")
    print("• Reduce manual data entry and processing errors")

    print("\n💡 Use Cases:")
    print("• Document processing and digitization")
    print("• Customer service automation")
    print("• Business intelligence and analytics")
    print("• Content management and organization")
    print("• Workflow automation and integration")


if __name__ == "__main__":
    main()