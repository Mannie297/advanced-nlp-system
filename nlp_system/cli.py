import argparse
import sys
from typing import Optional

from nlp_system.core.text_processor import TextProcessor
from nlp_system.core.sentiment_analyzer import SentimentAnalyzer
from nlp_system.core.entity_recognizer import EntityRecognizer
from nlp_system.core.topic_modeler import TopicModeler


def main(args: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Advanced NLP System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Text processing command
    process_parser = subparsers.add_parser("process", help="Process text")
    process_parser.add_argument("text", help="Text to process")
    process_parser.add_argument(
        "--language", default="en", help="Language of the text (default: en)"
    )

    # Sentiment analysis command
    sentiment_parser = subparsers.add_parser("sentiment", help="Analyze sentiment")
    sentiment_parser.add_argument("text", help="Text to analyze")
    sentiment_parser.add_argument(
        "--model",
        default="default",
        help="Sentiment analysis model to use (default: default)",
    )

    # Entity recognition command
    entity_parser = subparsers.add_parser("entities", help="Recognize entities")
    entity_parser.add_argument("text", help="Text to analyze")
    entity_parser.add_argument(
        "--types",
        nargs="+",
        default=["PERSON", "ORG", "GPE"],
        help="Entity types to recognize (default: PERSON ORG GPE)",
    )

    # Topic modeling command
    topic_parser = subparsers.add_parser("topics", help="Extract topics")
    topic_parser.add_argument("text", help="Text to analyze")
    topic_parser.add_argument(
        "--num-topics", type=int, default=5, help="Number of topics to extract"
    )

    args = parser.parse_args(args)

    if args.command == "process":
        processor = TextProcessor()
        result = processor.process(args.text, language=args.language)
        print("Processed text:", result)

    elif args.command == "sentiment":
        analyzer = SentimentAnalyzer()
        result = analyzer.analyze(args.text, model=args.model)
        print("Sentiment analysis:", result)

    elif args.command == "entities":
        recognizer = EntityRecognizer()
        result = recognizer.recognize(args.text, entity_types=args.types)
        print("Recognized entities:", result)

    elif args.command == "topics":
        modeler = TopicModeler()
        result = modeler.extract_topics(args.text, num_topics=args.num_topics)
        print("Extracted topics:", result)

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main()) 