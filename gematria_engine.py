class GematriaCipher:
    def __init__(self):
        self.mapping = {}

    def add_mapping(self, char, value):
        self.mapping[char] = value

    def encode(self, text):
        return sum(self.mapping.get(char, 0) for char in text)

    def decode(self, value):
        result = ""
        for char, char_value in self.mapping.items():
            while value >= char_value:
                value -= char_value
                result += char
        return result

    def clear_mapping(self):
        self.mapping.clear()

    def get_mapping(self):
        return self.mapping.copy()

    def remove_mapping(self, char):
        if char in self.mapping:
            del self.mapping[char]
            return True
        return False

    def has_mapping(self, char):
        return char in self.mapping

    def update_mapping(self, char, value):
        if char in self.mapping:
            self.mapping[char] = value
            return True
        return False

    def reset_mapping(self, new_mapping):
        self.mapping = new_mapping.copy()

class GematriaEngine:
    def __init__(self):
        self.cipher = GematriaCipher()

        english_extended_mapping = {
            'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9,
            'J': 10, 'K': 20, 'L': 30, 'M': 40, 'N': 50, 'O': 60, 'P': 70, 'Q': 80,
            'R': 90, 'S': 100, 'T': 200, 'U': 300, 'V': 400, 'W': 500, 'X': 600,
            'Y': 700, 'Z': 800, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9,
            'j': 10, 'k': 20, 'l': 30, 'm': 40, 'n': 50, 'o': 60, 'p': 70, 'q': 80,
            'r': 90, 's': 100, 't': 200, 'u': 300, 'v': 400, 'w': 500, 'x': 600,
            'y': 700, 'z': 800,
            '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '0': 0
        }

        self.cipher.reset_mapping(english_extended_mapping)

    def add_mapping(self, char, value):
        self.cipher.add_mapping(char, value)

    def encode(self, text):
        return self.cipher.encode(text)

    def decode(self, value):
        return self.cipher.decode(value)

    def clear_mapping(self):
        self.cipher.clear_mapping()

    def get_mapping(self):
        return self.cipher.get_mapping()

    def remove_mapping(self, char):
        return self.cipher.remove_mapping(char)

    def has_mapping(self, char):
        return self.cipher.has_mapping(char)

    def update_mapping(self, char, value):
        return self.cipher.update_mapping(char, value)

    def reset_mapping(self, new_mapping):
        self.cipher.reset_mapping(new_mapping)

    def numerological_reduction_path(self, text):
        path = []
        value = self.encode(text)
        while value > 9:
            path.append(value)
            value = sum(int(digit) for digit in str(value))
        path.append(value)
        return path

    def digital_root(self, text):
        path = self.numerological_reduction_path(text)
        return path[-1] if path else 0

    