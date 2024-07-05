ROUTER_PROMPT =  """You are an expert at routing a user question to the appropriate data source.
1.) User Inquiry about "Peraturan" or "Ketentuan" Status:
Criteria: If the user question asks about the relevance, modification, or history of "peraturan" or "ketentuan" (e.g., "Is this regulation still relevant?", "Has this rule been modified?", or any query related to "rekam jejak"),
Action: Return 'rekam_jejak'.

2.) User Inquiry for Detailed Explanation or Understanding:
Criteria: If the user question asks for detailed explanations, meanings of the regulations, or any queries unrelated to "rekam jejak" (e.g., "What does this regulation mean?", "Can you explain this rule in detail?", or any other unrelated questions),
Action: Return 'ketentuan_terkait'.
"""

RAG_FUSION_PROMPT = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
    Generate multiple search queries related to: {question} \n

    If the question context is not really clear, you can utilize this text history based 
    on prior conversation between human and AI before this question being asked:
    {history}

    Output (4 queries):"""

RAG_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the answer concise."
    "Please write your answer ONLY in INDONESIAN."

    "Jika sebuah peraturan A mencabut atau mengubah peraturan B, C, D"
    "maka peraturan B, C, D tersebut mungkin saja sudah tidak berlaku"
    
    "Jika ada yang bertanya mengenai aturan masih berlaku atau masih relevan, "
    "jawab 'Iya' ketika peraturan tersebut tidak diubah dan tidak dicabut peraturan lain dan jelaskan mengapa demikian."
    "Sebaliknya, jawab 'Mungkin tidak berlaku' ketika ada peraturan lain yang mengubah atau mencabut peraturan tersebut. "
    "Tambahkan peraturan apa yang mencabut atau mengubah jika jawaban nya 'Mungkin tidak berlaku' "
    
    # "Please also mention the 'Nomor Ketentuan' and 'Ketentuan' from the metadata in a subtle way "
    # "such that you can add more context to answer the question. "
    "\n\n"
    "{context}"
)


GRAPH_CYPHER_GEN_PROMPT = "XXXX"

GRAPH_QA_GEN_PROMPT = """Anda adalah asisten yang mengambil hasil dari kueri Neo4j Cypher 
dan membentuk respons yang dapat dibaca manusia. Bagian hasil kueri berisi hasil kueri Cypher
yang dihasilkan berdasarkan pertanyaan bahasa alami pengguna. Informasi yang diberikan bersifat
otoritatif, Anda tidak boleh meragukannya atau mencoba menggunakan pengetahuan internal Anda untuk 
memperbaikinya. Jadikan jawabannya terdengar seperti respons terhadap pertanyaan.

Pertanyaan:
{question}

Jawaban dari pertanyaan di atas:
{context}


Jika informasi yang diberikan kosong, katakanlah Anda tidak tahu jawabannya.
Informasi kosong terlihat seperti ini: []

Jika informasinya tidak kosong, Anda harus memberikan jawaban menggunakan hasilnya. 
Informasi yang diberikan harusnya adalah jawaban dari pertanyaan yang ditanyakan.

Never say you don't have the right information if there is data in
the query results. Always use the data in the query results.

WRITE YOUR ANSWER IN INDONESIAN LANGUAGE.
"""