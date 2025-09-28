from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.models import QueryResponse
from app.config import settings

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
                Use the job data provided. Do not hypothesize anything.
                If the query is not related to a job data search, say that it's not your expertise.
                If no job is returned or no returned job matches the user's query, apolozize and say that such type of job doesn't exist in the database.

    User query:
    "{query}"

    Job data:
    "{results}"

    ### Instructions:

    1. **Job Selection**: From the list of job data, identify the top 3 jobs that are most relevant to the user's query.
    - If the query includes terms like "remote only", "3+ years of experience", or "roles in USA", try to match the selected jobs match these conditions.

    2. **Explanation**:
    - For each selected job, provide a 1-2 sentence(max 75 characters) explanation for why it was chosen.
    ]
    '''
            )
        ]
    ).with_structured_output(QueryResponse)

    chain = prompt | llm  

    response = chain.invoke({'query': query, 'results': results})

    return response