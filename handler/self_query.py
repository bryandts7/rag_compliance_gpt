from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

from database.database import rekam_jejak_vector, ketentuan_terkait_vector
from utils.azure_openai import azure_llm

metadata_field_info = [
    AttributeInfo(
        name="Jenis Ketentuan",
        description="Jenis peraturan atau ketentuan",
        type="string",
    ),
    AttributeInfo(
        name="Judul Ketentuan",
        description="Judul peraturan atau ketentuan",
        type="string",
    ),
    AttributeInfo(
        name="Ketentuan",
        description="Pasal atau ketentuan spesifik dalam peraturan",
        type="string",
    ),
    AttributeInfo(
        name="Kodifikasi Ketentuan",
        description="Kategori kodifikasi peraturan",
        type="string",
    ),
    AttributeInfo(
        name="Nomor Ketentuan",
        description="Nomor referensi peraturan",
        type="string",
    ),
    AttributeInfo(
        name="Referensi",
        description="Referensi terkait peraturan",
        type="string",
    ),
    AttributeInfo(
        name="Tanggal Ketentuan",
        description="Tanggal ketika peraturan diterbitkan",
        type="string",
    ),
]

document_content_description = "Isi Ketentuan dari Peraturan"

from langchain_community.llms import Cohere

llm = Cohere(temperature=0)

def self_retriever_rekam_jejak():
    retriever = SelfQueryRetriever.from_llm(
        llm, rekam_jejak_vector(), document_content_description, metadata_field_info, verbose=True
    )
    return retriever

def self_retriever_ketentuan():
    retriever = SelfQueryRetriever.from_llm(
        llm, ketentuan_terkait_vector(), document_content_description, metadata_field_info, verbose=True
    )
    return retriever

