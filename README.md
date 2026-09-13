# Phonebot AI Engineer Technical Challenge

## Overview

This is a technical challenge for the Phonebot AI Engineer candidates. You will build a post-processing pipeline that extracts caller information from AI phone bot recordings.

## Business Problem

Our AI phone bots conduct conversations with callers to collect contact information. After each call, we need to extract structured data from the recordings for downstream processing (updating the cases, CRM, follow-ups, etc.).

Your task is to build a pipeline that processes audio recordings and extracts the following entities:
- **first_name**: Caller's first name
- **last_name**: Caller's last name
- **email**: Caller's email address
- **phone_number**: Caller's phone number

## Technical Challenge

Build a system that:
1. Processes the provided audio recordings
2. Extracts the four entity types listed above
3. Outputs results in a format that can be compared against the ground truth

### Requirements
- Maximize accuracy against the provided ground truth
- Follow best practices for AI/ML pipelines
- Consider production readiness in your design

## Evaluation Criteria

We evaluate submissions based on the following criteria:

### 1. Accuracy
How well does your system perform against the ground truth? We'll measure:
- Per-entity accuracy (first_name, last_name, email, phone_number)
- Overall extraction accuracy
- Handling of edge cases (special characters, international names, etc.)

### 2. AI Engineering Approach
- **Transcription**: Model choice and configuration
- **Prompting Strategy**: How you structure prompts for extraction
- **Error Handling**: Graceful degradation, retry logic, fallbacks

### 3. Future-proofing & Controllability
- **Monitoring**: How would you track system performance?
- **Observability**: Logging, tracing, debugging capabilities
- **Prompt Management**: Easy updates without code changes
- **A/B Testing**: Ability to compare different approaches

### 4. Code Quality
- Clean, readable, and maintainable code
- Appropriate documentation
- Good project structure
- Test coverage where appropriate

## Sample Data

The `data/` directory contains:
- `recordings/`: 30 WAV audio files (call_01.wav through call_30.wav)
- `ground_truth.json`: Expected extraction results for each recording

### Ground Truth Format

```json
{
  "recordings": [
    {
      "id": "call_01",
      "file": "call_01.wav",
      "expected": {
        "first_name": "Jürgen",
        "last_name": "Meyer",
        "email": "j.meyer@gmail.com",
        "phone_number": "+4917284492"
      }
    }
  ]
}
```

**Note**: Some entities may have multiple acceptable values (stored as arrays), e.g., `"first_name": ["Lisa Marie", "Lisa-Marie"]`. Your extraction is considered correct if it matches any of the acceptable values.

## Important Notes

- **Language**: All recordings are in German, but German language proficiency is NOT required for this challenge. The focus is on your engineering approach, not language skills.
- **Time**: There is no strict time limit, but we expect a reasonable solution within a few days.
- **Resources**: You may use any tools, libraries, or APIs you see fit.

## Discussion Questions

Be prepared to discuss the following during the technical interview:

1. **Production Monitoring**: How would you monitor and improve the system in production?

2. **Extensibility**: How would you handle new entity types (e.g., address, company name)?

3. **System Health**: What metrics would you track to measure system health?

## Delivery Format

Please prepare:
1. **Working Code/Prototype**: A functional implementation that processes the recordings
2. **Brief Documentation**: How to run your solution and any design decisions
3. **1-Hour Technical Discussion**: We'll review your solution together and discuss the questions above

## Getting Started

1. Review the sample recordings and ground truth
2. Design your extraction pipeline
3. Implement and test your solution
4. Document your approach and any trade-offs made

Good luck!
