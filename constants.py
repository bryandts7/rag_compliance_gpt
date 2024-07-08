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

    Output (3 queries):"""

RAG_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know."
    "Please write your answer ONLY in INDONESIAN."
    
    # "Jika sebuah peraturan A mencabut atau mengubah peraturan B, C, D"
    # "maka peraturan B, C, D tersebut mungkin saja sudah tidak berlaku"
    
    # "Jika ada yang bertanya mengenai aturan masih berlaku atau masih relevan, "
    # "jawab 'Iya' ketika peraturan tersebut tidak diubah dan tidak dicabut peraturan lain dan jelaskan mengapa demikian."
    # "Sebaliknya, jawab 'Mungkin tidak berlaku' ketika ada peraturan lain yang mengubah atau mencabut peraturan tersebut. "
    # "Tambahkan peraturan apa yang mencabut atau mengubah jika jawaban nya 'Mungkin tidak berlaku' "

    "Jika konteks yang diberikan tidak relevan dengan pertanyaan, jangan pernah membuat jawaban dari pengetahuanmu sendiri"
    "Jawab kalau anda tidak tahu jika konteks yang diberikan tidak sesuai dengan pertanyaan."
    
    # "Please also mention the 'Nomor Ketentuan' and 'Ketentuan' from the metadata in a subtle way "
    # "such that you can add more context to answer the question. "
    "\n\n"
    "{context}"
)

RAG_REKAM_JEJAK_PROMPT = (
    "You will be provided with two sources of context: one from GraphRAG and one from RAG (which contains several Documents)."
    "Your task is to combine the results from these two contexts effectively. However, prioritize the context from GraphRAG if it is not empty."
    "If the context from GraphRAG is empty, then you should retrieve fully from the context provided by RAG."

    "If GraphRAG context provides information about a topic but is missing some details, supplement it with additional details from the RAG context."
    
    "GraphRAG Context:"
    "{structured}"
    "If the above context is empty, you can rely on the Documents retreived here as the secondary context:"
    "{unstructured}"
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

EVAL_QUESTIONS = [
    "Apa batas waktu yang diberikan untuk persetujuan atau penolakan atas permohonan izin usaha setelah dokumen permohonan diterima secara lengkap?",
    "Berapa lama waktu yang diberikan bagi bank yang telah mendapat izin usaha dari Gubernur Bank Indonesia untuk memulai kegiatan usaha perbankan?",
    "Siapa yang dapat mendirikan dan memiliki Bank Perantara?",
    "Apa yang terjadi jika dokumen permohonan izin usaha tidak lengkap?",
    "Apa saja yang termasuk dalam Bank Pelapor?",
    "Apa saja jenis data yang wajib dilaporkan dalam Laporan Harian Bank Umum (LHBU)?",
    "Apa yang harus dilakukan jika penyampaian dan/atau koreksi LHBU dilakukan setelah batas waktu?",
    "Apa itu Data JIBOR dan bagaimana penetapannya oleh Bank Indonesia?",
    "Berapa persentase minimal Dana Pendidikan dari Anggaran Pengeluaran Sumber Daya Manusia mulai tahun 2003?",
    "Apa yang harus dilakukan Bank jika terdapat sisa Dana Pendidikan?",
    "Apa hak bagi pihak yang membeli saham Bank Perantara?",
    "Berapa lama pihak yang membeli saham Bank Perantara dapat memiliki saham Bank melebihi batas maksimum kepemilikan saham?",
    "Apa yang dimaksud dengan penerapan manajemen risiko dalam konteks Rencana Bisnis Bank Umum?",
    "Apa saja yang harus dicakup dalam uraian kinerja keuangan Bank Umum?",
    "Apa yang diuraikan dalam realisasi pemberian kredit kepada Usaha Mikro, Kecil, dan Menengah (UMKM)?"

]

EVAL_ANSWERS = [
    "Batas waktu yang diberikan adalah paling lambat 60 (enam puluh) hari kerja setelah dokumen permohonan diterima secara lengkap.",
    "Bank wajib melakukan kegiatan usaha perbankan paling lambat 60 (enam puluh) hari kerja terhitung sejak tanggal izin usaha diterbitkan.",
    "Bank Perantara hanya dapat didirikan dan dimiliki oleh Lembaga Penjamin Simpanan (LPS).",
    "Persetujuan atau penolakan tidak akan diproses hingga dokumen permohonan diterima secara lengkap.",
    "Bank Pelapor terdiri dari: 1.) Kantor pusat Bank Umum konvensional\n 2.)Kantor pusat Bank Umum Syariah\n 3.) Kantor pusat Bank Pembangunan Daerah\n 4.)Kantor pusat Bank Umum yang berkedudukan di luar negeri (cabang)",
    "Jenis data yang wajib dilaporkan dalam LHBU meliputi data transaksional dan non-transaksional.",
    "Jika penyampaian dan/atau koreksi LHBU dilakukan setelah batas waktu yang ditentukan, Bank harus mengikuti prosedur khusus untuk pengajuan keterlambatan dan memastikan bahwa semua koreksi dibuat dengan benar.",
    "Data Jakarta Interbank Offered Rate (JIBOR) ditetapkan oleh Bank Indonesia sebagai acuan suku bunga antar bank. Bank Indonesia menetapkan Data JIBOR berdasarkan laporan yang diterima dari bank-bank pelapor.",
    "Sekurang-kurangnya sebesar 5'%' dari Anggaran Pengeluaran Sumber Daya Manusia.",
    "Bank wajib menyetorkan Dana Pendidikan tersebut kepada Institut Bankir Indonesia (IBI) untuk digunakan sebagai biaya pendidikan perbankan atau menambahkan Dana Pendidikan tersebut ke Dana Pendidikan tahun berikutnya.",
    "Pihak yang membeli saham Bank Perantara yang telah dijual dapat memiliki saham Bank melebihi batas maksimum kepemilikan saham sebagaimana pemegang saham yang memiliki Bank dalam penanganan atau penyelamatan oleh LPS sebagaimana diatur dalam ketentuan OJK mengenai kepemilikan saham bank umum.",
    "Pihak yang membeli saham Bank Perantara dapat memiliki saham Bank melebihi batas maksimum kepemilikan saham paling lama 20 (dua puluh) tahun sejak pembelian saham Bank Perantara dari LPS.",
    "Penerapan manajemen risiko dalam konteks Rencana Bisnis Bank Umum mencakup penjelasan kuantitatif dan kualitatif mengenai kondisi Bank pada saat penyusunan Rencana Bisnis. Ini termasuk evaluasi penerapan manajemen risiko, profil risiko, dan efektivitas serta hasil penerapan ketentuan terkait Anti Pencucian Uang dan Pencegahan Pendanaan Terorisme (APU dan PPT) serta fungsi kepatuhan Bank.",
    "Uraian kinerja keuangan Bank Umum harus mencakup:\n 1. Hasil pelaksanaan rencana tindak (action plan) untuk memperbaiki kinerja Bank.\n 2. Kinerja permodalan, termasuk kecukupan, komposisi, dan kemampuan modal Bank untuk mengcover risiko, mendukung pertumbuhan usaha, dan akses kepada sumber permodalan.\n 3. Kinerja rentabilitas, termasuk pencapaian ROA, ROE, NIM, perkembangan laba operasional, rasio BOPO, dan rasio beban operasional selain bunga terhadap pendapatan kegiatan utama.",
    "Uraian mengenai realisasi pemberian kredit mencerminkan peranan Bank Umum dalam mendukung perkembangan UMKM, dengan pengelompokan UMKM mengacu pada kriteria usaha berdasarkan Undang-Undang yang mengatur mengenai usaha mikro, kecil, dan menengah."

]