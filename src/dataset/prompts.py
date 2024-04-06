prompts_dict = {
 "MEDICAL": """You are an exceptionally erudite Named Entity Recognition (NER) extractor, specialised in MEDICAL DOMAIN. I will provide you list of the entities you need to extract, the MEDICAL CASE REPORT from where your extract the entities and the OUTPUT FORMAT.

INSTRUCTIONS:

    1. Pick sentences one by one from the given INPUT MEDICAL CASE REPORT
    2. EXTRACT ALL POSSIBLE NAMED ENTITIES FROM ENTITY TYPES DEFINED, MENTIONED IN THE SENTENCE PICKED
    3. Keep all the named entities EXACTLY SAME as present in the MEDICAL CASE REPORT, EACH ENTITY EXTRACTED SHOULD BE FOUND AS IT IS IN THE SENTENCE. 

    NAMED ENTITIES:
    ['AGE', 'SEX', 'HISTORY', 'DISEASE_DISORDER', 'CLINICAL_EVENT', 'SIGN_SYMPTOM', 'DETAILED_DESCRIPTION', 'DURATION', 'LAB_VALUE', 'DIAGNOSTIC_PROCEDURE', 'MEDICATION', 'BIOLOGICAL_STRUCTURE', 'DISTANCE', 'DATE', 'NONBIOLOGICAL_LOCATION', 'QUALITATIVE_CONCEPT', 'THERAPEUTIC_PROCEDURE', 'FREQUENCY', 'OUTCOME', 'ACTIVITY', 'SEVERITY', 'SUBJECT', 'QUANTITATIVE_CONCEPT', 'VOLUME', 'ADMINISTRATION', 'AREA', 'COREFERENCE', 'PERSONAL_BACKGROUND', 'OTHER_EVENT', 'DOSAGE', 'COLOR', 'MASS', 'TEXTURE', 'SHAPE', 'OCCUPATION', 'FAMILY_HISTORY', 'BIOLOGICAL_ATTRIBUTE', 'TIME', 'HEIGHT', 'WEIGHT', 'OTHER_ENTITY']
    
    JSON OUTPUT FORMAT FOR REFERENCE:
    [
        {{
            "SENTENCE": "sentenc1 from MEDICAL CASE REPORT",
            "ENTITIES": 
            {{
                "ENTITY TYPE": [list of ENTITY from sentence1],
                "ENTITY TYPE": [list of ENTITY from sentence1],
                 ...
            }}
       }}, 
        {{
            "SENTENCE": "sentenc2 from MEDICAL CASE REPORT",
            "ENTITIES": 
            {{
                 "ENTITY TYPE": [list of ENTITY from sentence2],
                 "ENTITY TYPE": [list of ENTITY from sentence2],
                  ....
           }}
       }}
    ]

    IMPORTANT NOTES:
    - STRICTLY FOLLOW THE FORMAT SPECIFIED IN THE ABOVE JSON OUTPUT FORMAT FOR REFERENCE AS IT WILL BE USED BY OTHER COMPUTER PROGRAMS
    - DO NOT PROVIDE ANY EXTRA EXPLANATION OR INFORMATION.
    - GO THROUGH "EACH" AND "EVERY SENTENCE" STEP BY STEP AS MENTIONED IN THE INSTRUCTIONS AND ONLY WRITE ENTITIES EXTRACTED FROM CORRESPONDING SENTENCE

    INPUT MEDICAL CASE REPORT: input_____
    Here are the extracted entities from the given MEDICAL CASE REPORT:""",
    
    "MEDICAL_DEFINITIONS": '''YOU ARE A MEDICAL PROFESSIONAL UNDERSTANDING ALL NUANCES OF CASE REPORTS
Your job is to extract Named Entities, specific in MEDICAL DOMAIN. I will provide you list of the entities you need to extract, the MEDICAL CASE REPORT from where your extract the entities and the OUTPUT FORMAT.

INSTRUCTIONS:

    1. Pick sentences one by one from the given INPUT MEDICAL CASE REPORT
    2. EXTRACT ALL POSSIBLE NAMED ENTITIES FROM ENTITY TYPES DEFINED, MENTIONED IN THE SENTENCE PICKED
    3. Keep all the named entities EXACTLY SAME as present in the MEDICAL CASE REPORT, EACH ENTITY EXTRACTED SHOULD BE FOUND AS IT IS IN THE SENTENCE. 
   
   NAMED ENTITIES:
   Here's a guideline for annotating the named entities from case reports:

- PATIENT_AGE: Numeric value indicating the patient's age at the time of the event or diagnosis.
- PATIENT_SEX: Biological sex of the patient, specified as female, male, or intersex. Gender identity is included if relevant and available.
- MEDICAL_HISTORY: Past personal health history, detailing previous diseases, surgeries, and significant health events.
- FAMILY_MEDICAL_HISTORY: Detailed medical history of the patient's family, including genetic conditions and hereditary diseases.
- LIFESTYLE_FACTORS: Patient's physical, occupational, and leisure activities that influence or signify their health status.
- COMORBIDITY_DETAILS: Information on additional diagnoses, highlighting their influence on the primary condition and its management.
- PREV_SURGICAL_INTERVENTIONS: History of surgical procedures, noting types, dates, and outcomes relevant to current health.
- ENVIRONMENTAL_FACTORS: Environmental conditions and exposures with potential relevance to the patient's condition.
- SOCIOECONOMIC_CONTEXT: The patient's economic, educational, and occupational background affecting their health access and decisions.
- CULTURAL_BACKGROUND: The patient's self-described ethnic or cultural identity, considering its relevance to healthcare outcomes.
- KNOWN_ALLERGIES: Documented allergies or hypersensitivities, including substances and response details.
- OCCUPATIONAL_STATUS: Current or recent job role and its physical or psychological implications on the patient’s health.
- DIETARY_HABITS: Description of the patient's diet, noting special diets, restrictions, and preferences affecting health.
- BEHAVIOR_AND_MENTAL_WELLNESS: Lifestyle choices related to smoking, alcohol, exercise, and documented mental health conditions.
- PHYSICAL_STATS: Physical attributes such as height, weight, and body mass index at the time of reporting. Extract the complete attribute and their values.
- CURRENT_MEDICATIONS: List of medications prescribed to the patient at the time of the case, including dosage, frequency, and administration route.
- DIAGNOSED_CONDITION: The specific illness or medical disorder diagnosed in the patient, distinguishing primary condition from comorbidities.
- SIGNIFICANT_HEALTH_EVENTS: Critical health-related incidents like acute episodes, interventions, or surgeries relevant to the patient's care.
- SYMPTOM_DETAILS: Description of symptoms experienced by the patient, covering both subjective feelings and objective signs. Non-numeric assessment of how symptoms affect the patient’s daily life and quality of life, Regularity and frequency, duration of symptoms appearing over a specific period.
- AFFECTED_SYSTEM: Body system, organ, or area directly involved in or affected by the diagnosed condition.
- CASE_SETTING: Geographic and healthcare facility details where the case occurred and was managed. Include the type of facility, location, and care providers involved.
- DIAGNOSTIC_PROCEDURES: Detailed information on diagnostic tests performed, including methodology, findings, and conclusions.
- LAB_RESULTS: Numeric outcomes and relevant notes from diagnostic lab tests, with deviations from normal ranges highlighted.
- CASE_MEASUREMENTS: Specific numerical data related to the case, encompassing lab values, measurements, and medication doses.
- POST_TREATMENT_STATUS: Description of patient's status following treatment, including recovery level, complications, or ongoing problems.
- MANAGEMENT_APPROACH: Comprehensive strategy used to manage the patient's condition, including medications, therapies, and lifestyle changes.
- ADMINISTRATION_DETAILS: Comprehensive details on the delivery methods for therapies and medications.
- IMAGING_FINDINGS: Interpretations of radiological and other imaging results, specifying types and key conclusions.
- TREATMENT_OUTCOMES: Description of any adverse effects or complications from treatments or disease progression, including responses to these events.

    JSON OUTPUT FORMAT FOR REFERENCE:
    {{
        "response": [
            {{
                "SENTENCE": "sentenc1 from MEDICAL CASE REPORT",
                "ENTITIES": 
                {{
                    "ENTITY TYPE": [list of ENTITY from sentence1],
                    "ENTITY TYPE": [list of ENTITY from sentence1],
                    ...
                }}
        }}, 
            {{
                "SENTENCE": "sentenc2 from MEDICAL CASE REPORT",
                "ENTITIES": 
                {{
                    "ENTITY TYPE": [list of ENTITY from sentence2],
                    "ENTITY TYPE": [list of ENTITY from sentence2],
                    ....
            }}
        }}
        ]
    }}
    INPUT MEDICAL CASE REPORT: ```input_____```
IMPORTANT NOTES:
    - STRICTLY FOLLOW THE FORMAT SPECIFIED IN THE ABOVE JSON OUTPUT FORMAT FOR REFERENCE AS IT WILL BE USED BY OTHER COMPUTER PROGRAMS
    - DO NOT PROVIDE ANY EXTRA EXPLANATION OR INFORMATION.
    - GO THROUGH "EACH" AND "EVERY SENTENCE" THOROUGHLY, THINK CAREFULLY AND ONLY WRITE ALL ENTITIES EXTRACTED FROM CORRESPONDING SENTENCE
    - EXTRACT A COMPLETE MEANINGFUL ENTITY, INCLUDE ABBREVIATIONS IF THEY ARE PART OF THE ENTITY
    Here are the extracted entities from the given MEDICAL CASE REPORT:
}'''
    

}
