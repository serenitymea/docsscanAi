import argparse
from rag_system import RAGSystem


def main():
    """Main function for CLI interface"""
    parser = argparse.ArgumentParser(description='RAG system for documents')
    parser.add_argument('--voyage-key', required=True, help='Voyage AI API key')
    parser.add_argument('--gemini-key', required=True, help='Google Gemini API key')
    parser.add_argument('--db-path', default='./chroma_db', help='Database path')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    add_parser = subparsers.add_parser('add', help='Add document')
    add_parser.add_argument('file_path', help='Document file path')
    add_parser.add_argument('--name', help='Document name in database')
    
    ask_parser = subparsers.add_parser('ask', help='Ask question')
    ask_parser.add_argument('question', help='Your question')
    ask_parser.add_argument('--results', type=int, default=3, help='Number of fragments to search')

    subparsers.add_parser('stats', help='Show knowledge base statistics')

    subparsers.add_parser('interactive', help='Interactive mode')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return

    try:
        rag = RAGSystem(args.voyage_key, args.gemini_key, args.db_path)
    except Exception as e:
        print(f"initialization error: {str(e)}")
        return

    if args.command == 'add':
        try:
            success = rag.add_document(args.file_path, args.name)
            if success:
                print("document successfully added!")
            else:
                print("failed to add document")
        except Exception as e:
            print(f"error adding document: {str(e)}")
    
    elif args.command == 'ask':
        try:
            result = rag.ask_question(args.question, args.results)
            print("\n" + "="*60)
            print("ANSWER:")
            print(result['answer'])
            print("\n" + "="*60)
            print("SOURCES:")
            for source in result['sources']:
                print(f"  • {source['document']} (relevance: {source['similarity']})")
            print(f"\nfragments used: {result['chunks_used']}")
        except Exception as e:
            print(f"error processing question: {str(e)}")
    
    elif args.command == 'stats':
        try:
            stats = rag.get_stats()
            print("\nKNOWLEDGE BASE STATISTICS:")
            print(f"  • total fragments: {stats['total_chunks']}")
            print(f"  • total documents: {stats['total_documents']}")
            print(f"  • documents {', '.join(stats['documents']) if stats['documents'] else 'none'}")
        except Exception as e:
            print(f"Error getting statistics: {str(e)}")
    
    elif args.command == 'interactive':
        print("\nInteractive RAG system mode")
        print("commands 'add <file>' - add document, 'stats' - statistics, 'quit' - exit")
        print("or just ask questions\n")
        
        while True:
            try:
                user_input = input("\nyour input: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    break
                
                elif user_input.startswith('add '):
                    file_path = user_input[4:].strip()
                    if file_path:
                        success = rag.add_document(file_path)
                        if success:
                            print("Document added successfully")
                        else:
                            print("Failed to add document")
                    else:
                        print("Please specify file path")
                
                elif user_input.lower() == 'stats':
                    stats = rag.get_stats()
                    print(f"\nfragments {stats['total_chunks']}, documents {stats['total_documents']}")
                
                elif user_input:
                    result = rag.ask_question(user_input)
                    print(f"\n{result['answer']}")
                    if result['sources']:
                        print(f"sources: {', '.join(s['document'] for s in result['sources'])}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"error {str(e)}")
        
        print("\nbb")


if __name__ == "__main__":
    main()