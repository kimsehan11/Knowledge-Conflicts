import os
from typing import List, Dict, Any
from tavily import TavilyClient
import json


class TavilyRAG:

    client = None

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY 설정 필요함")

        self.client = TavilyClient(api_key=self.api_key)

    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth="basic", 
            )

            return response

        except Exception as e:
            print(f"검색 중 오류 발생: {e}")
            return {}

    def extract_documents(self, search_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        검색 결과에서 문서 정보 추출

        Args:
            search_results: Tavily API 검색 결과

        Returns:
            문서 리스트 (title, content, url 포함)
        """
        documents = []

        if "results" in search_results:
            for result in search_results["results"]:
                doc = {
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "url": result.get("url", ""),
                    "score": result.get("score", 0.0)
                }
                documents.append(doc)

        return documents

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, str]]:
        """
        쿼리에 대한 상위 K개 문서 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 수

        Returns:
            상위 K개 문서 리스트
        """
        search_results = self.search(query, max_results=top_k)
        documents = self.extract_documents(search_results)

        return documents

    def format_context(self, documents: List[Dict[str, str]]) -> str:
        """
        검색된 문서들을 컨텍스트 문자열로 포맷팅

        Args:
            documents: 문서 리스트

        Returns:
            포맷팅된 컨텍스트 문자열
        """
        context_parts = []

        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[문서 {i}]")
            context_parts.append(f"제목: {doc['title']}")
            context_parts.append(f"내용: {doc['content']}")
            context_parts.append(f"출처: {doc['url']}")
            context_parts.append("")  # 빈 줄

        return "\n".join(context_parts)

    def rag_pipeline(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        전체 RAG 파이프라인 실행

        Args:
            query: 사용자 질문
            top_k: 검색할 문서 수

        Returns:
            검색된 문서와 컨텍스트를 포함한 딕셔너리
        """
        # 문서 검색
        documents = self.retrieve(query, top_k=top_k)

        # 컨텍스트 생성
        context = self.format_context(documents)

        return {
            "query": query,
            "documents": documents,
            "context": context,
            "num_documents": len(documents)
        }


def main():
    """사용 예시"""
    # Tavily RAG 초기화
    rag = TavilyRAG()

    # 검색 쿼리
    query = "What is the capital of France?"

    # RAG 파이프라인 실행
    result = rag.rag_pipeline(query, top_k=10)

    # 결과 출력
    print(f"쿼리: {result['query']}")
    print(f"검색된 문서 수: {result['num_documents']}")
    print("\n=== 검색된 문서 ===")

    for i, doc in enumerate(result['documents'], 1):
        print(f"\n[문서 {i}]")
        print(f"제목: {doc['title']}")
        print(f"점수: {doc['score']}")
        print(f"URL: {doc['url']}")
        print(f"내용: {doc['content'][:200]}...")  # 처음 200자만 출력

    print("\n=== 전체 컨텍스트 ===")
    print(result['context'])

    # 결과를 JSON 파일로 저장
    with open("tavily_rag_output.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n결과가 tavily_rag_output.json 파일로 저장되었습니다.")


if __name__ == "__main__":
    main()
