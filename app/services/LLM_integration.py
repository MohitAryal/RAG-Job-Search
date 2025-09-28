from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.config import settings

def llm_result(results, query):
    llm = ChatGroq(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        groq_api_key=settings.llm_api_key
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                '''You are a highly skilled job search assistant. Your task is to extract and summarize the most relevant job listings for the user based on their query and provided job data.
                Use the job data provided. Do not hypothesize anything.
                If the query is not related to a job data search, be sincere and just return that it's not your expertise(no need to adhere to the output format for this case).
                If no job is returned or no returned job matches the user's query, apolozize and say that such type of job doesn't exist in the database.

    User query:
    "{query}"

    Job data:
    "{results}"

    ### Instructions:

    1. **Job Selection**: From the list of job data, identify the top 3 jobs that are most relevant to the user's query.
    - Carefully respect filters in the query such as location, remote preference, experience level, job title, or required skills if mentioned in the query.
    - If the query includes terms like "remote only", "3+ years of experience", or "roles in USA", make sure the selected jobs match these conditions. If this means there's no match, just apolozize and select the results without the filter.

    2. **Content Snippet Extraction**:
    - For each selected job, extract the 2–3 most relevant sentences (maximum of 400 characters in total per job) that strongly align with the query.
    - Prioritize mention of key elements from the query: skills, seniority level, location, remote work, or tech stack.
    - Keep the text concise and directly quoted from the job description. Do not paraphrase.

    3. **Explanation**:
    - For each selected job, provide a 1-sentence explanation for why it was chosen.

    4. **Output Format**:
    - Return a JSON list containing 3 items, each representing one selected job.
    - Each item should include the following fields:
        - `ID`: The id of relevant chosen job present in the provided Job data. Return exact job id provided.
        - `content_snippet`: Extracted text from the job description (quoted, ≤500 characters).
        - `explanation`: One sentence describing why the job is relevant to the query.
    ]
    '''
            )
        ]
    )

    chain = prompt | llm | JsonOutputParser()
    
    # Keys you want to keep
    keys_to_keep = {'ID', 'Job Title', 'Job Location', 'Job Level', 'Job Description'}

    # Extract only those keys from each dictionary
    filtered_result = [{k: d[k] for k in keys_to_keep if k in d} for d in results]   

    response = chain.invoke({'query': query, 'results': filtered_result})

    return response