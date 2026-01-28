import argparse
from rag_system import RAGSystem


def cmd_add(rag: RAGSystem, args):
    rag.add_document(args.file_path, args.name)
    print("Document added successfully")


def cmd_ask(rag: RAGSystem, args):
    result = rag.ask(args.question, args.results)

    print("\n" + "=" * 60)
    print("ANSWER:\n")
    print(result["answer"])

    if result["sources"]:
        print("\nSOURCES:")
        for s in result["sources"]:
            print(f" • {s['document']} (score: {s['similarity']})")

    print("=" * 60)


def cmd_stats(rag: RAGSystem, args):
    stats = rag.stats()

    print("\nKnowledge Base Stats")
    print(f"Chunks: {stats['chunks']}")
    print(f"Documents: {stats['documents']}")

    if stats.get("doc_list"):
        print("Files:", ", ".join(stats["doc_list"]))


def cmd_interactive(rag: RAGSystem, args):
    print("\nInteractive mode (type 'quit' to exit)\n")

    while True:
        user_input = input(">> ").strip()

        if user_input.lower() in ("quit", "exit"):
            break

        if user_input.startswith("add "):
            rag.add_document(user_input[4:].strip())
            print("Added")
            continue

        if user_input == "stats":
            cmd_stats(rag, args)
            continue

        if user_input:
            result = rag.ask(user_input)
            print("\n" + result["answer"])


def build_parser():
    parser = argparse.ArgumentParser("RAG Document QA System")

    parser.add_argument("--voyage-key", required=True)
    parser.add_argument("--gemini-key", required=True)
    parser.add_argument("--db-path", default="./chroma_db")

    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add")
    add.add_argument("file_path")
    add.add_argument("--name")

    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--results", type=int, default=3)

    sub.add_parser("stats")
    sub.add_parser("interactive")

    return parser


def main():
    args = build_parser().parse_args()

    rag = RAGSystem(
        voyage_api_key=args.voyage_key,
        gemini_api_key=args.gemini_key,
        db_path=args.db_path,
    )

    commands = {
        "add": cmd_add,
        "ask": cmd_ask,
        "stats": cmd_stats,
        "interactive": cmd_interactive,
    }

    commands[args.command](rag, args)


if __name__ == "__main__":
    main()
