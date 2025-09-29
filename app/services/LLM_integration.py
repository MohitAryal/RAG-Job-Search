from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.config import settings
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(
    model=settings.llm_model,
    temperature=settings.llm_temperature,
    max_tokens=settings.llm_max_tokens,
    groq_api_key=settings.llm_api_key
    )


def llm_result(results, query):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                '''You are a highly skilled job search assistant. Your task is to get the most relevant job listings for the user based on their query and provided job data.
    Use **only** the job data provided. Do not hypothesize anything.
    If the query is not related to a job data search, say that it's not your expertise.
    If no job is returned or no returned job matches the user's query,just apologize and say that such type of job doesn't exist in the database. No need to mention anything else.

    User query:
    "{query}"

    Job data:
    "{results}"

    ### Instructions:

    1. **Job Selection**: 
    From the list of job data, find the job that matches the user's query.
    - Match only if the job title and description clearly indicate a role in the same occupation or industry as the query.
    - Match the query with the title first. Move to other steps only if the user's intention in the query logically matches the job title clearly.
    - Validate that the job is in the same professional field or occupation category as the user's query.A "doctor" must relate to healthcare or medicine, not data, business, or analytics.If such alignment is not clear in the job title or description, do not select the job.
    - Do not select jobs based on keyword overlap alone. If the job does not clearly belong to the same professional domain, do not include it even if similar words are used.
    - If match is found, identify the top jobs that are most relevant to the user's query. Select a maximum of 3 relevant jobs. If there aren't 3 jobs relevant to the query, you may respond with any number of jobs you see fit.
    - If the query includes filters like "remote only", "3+ years of experience", or "roles in USA", try to match the selected jobs match these conditions.
    - If a job is not selected, do not mention it and do not give explanation for why it wasn't selected.

    2. **Explanation**:
    - For each selected job, provide a 2-3 sentence(max 100 characters) explanation for why it was chosen based on the chunks received in job data.
    - When explaining why a job was chosen, reference specific terms in the **job title** and **description** that justify the selection.
    - If no job is selected, explanation is not required.
    
    ### Output Format:
    If there is a match between job and query, keep the output format exactly as follows:

    1. **[Job ID] - [Job Title: exact job title quoted from metadata of Job Data]**  
    - This job was chosen because [reason]. [example: this job was chosen because the user mentioned so and so in the title.] If the title doesn't match, no need to give this reason
    - [Any other reason if applicable].  example: The location so and so mentioned by the user also matches.
    - [Any other reason if applicable]. example: This is shown because user mentioned 3 years experince. 
    - **Location:** [location: include all the location details mentioned in the metadata.]  
    - **Seniority:** [seniority]

    2. **[Job ID] - [Job Title]**
    ...

    3. **[Job ID] - [Job Title]**
    ...

    Else in case of no match return a one-sentence answer mentioning the one of the two reasons based on the query: 
    - The job is not present in the database if it's a job related query.
    - I am unable to handle tasks other than matching jobs if it's not a job related query.
    '''
            )
        ]
    )
    
    chain = prompt | llm  | StrOutputParser()

    response = chain.invoke({'query': query, 'results': results})

    return response