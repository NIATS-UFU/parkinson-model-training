import typer
from loguru import logger


def sum_two_ints(a: int, b: int) -> int:
    """Sum two integers"""
    return a + b


def main(salutation: str = "Hello!"):
    """Main function"""
    print(f"{salutation} World")
    logger.info(f"{salutation} World")


if __name__ == "__main__":
    typer.run(main)
