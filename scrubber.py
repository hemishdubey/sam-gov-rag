from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import re

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
CREDIT_CARD_PATTERN = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')

def scrub_text(text: str) -> tuple[str, list]:
    detected = []

    # Manual regex scrubbing first
    if SSN_PATTERN.search(text):
        text = SSN_PATTERN.sub('<US_SSN>', text)
        detected.append("US_SSN")

    if CREDIT_CARD_PATTERN.search(text):
        text = CREDIT_CARD_PATTERN.sub('<CREDIT_CARD>', text)
        detected.append("CREDIT_CARD")

    # Presidio for everything else
    results = analyzer.analyze(
        text=text,
        entities=[
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "IP_ADDRESS",
            "URL",
            "US_BANK_NUMBER",
            "CRYPTO",
            "US_DRIVER_LICENSE",
            "US_PASSPORT",
            "IBAN_CODE",
            "MEDICAL_LICENSE"
        ],
        language="en",
        score_threshold=0.3
    )

    if results:
        anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
        text = anonymized.text
        detected += list(set([r.entity_type for r in results]))

    return text, list(set(detected))

if __name__ == "__main__":
    test = "My name is John Smith, email me at john@company.com or call 555-123-4567. SSN: 123-45-6789"
    scrubbed, detected = scrub_text(test)
    print(f"Original:  {test}")
    print(f"Scrubbed:  {scrubbed}")
    print(f"Detected:  {detected}")