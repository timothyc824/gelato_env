from enum import Enum


class Flavour(Enum):
    VANILLA = "vanilla"
    CHOCOLATE = "chocolate"
    STRAWBERRY = "strawberry"
    MINT = "mint"
    COOKIES_AND_CREAM = "cookies_and_cream"
    COFFEE = "coffee"
    PISTACHIO = "pistachio"
    LEMON = "lemon"
    MANGO = "mango"
    RASPBERRY = "raspberry"

    @classmethod
    def get_all_flavours(cls):
        return [flavour.value for flavour in cls]

    @classmethod
    def get_flavour_encoding(cls):
        one_hot = {}
        for idx, flavour in zip(range(len(cls)), cls):
            one_hot[flavour.value] = idx
        return one_hot
