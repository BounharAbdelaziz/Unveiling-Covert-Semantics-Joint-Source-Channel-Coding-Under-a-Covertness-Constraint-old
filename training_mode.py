from enum import Enum

class TRAINING_MODE(Enum):
    FEATURE_EXTRACTOR = 1
    AUTO_ENCODER = 2
    TEST = 3
    TEST_FEATURE_EXTRACTOR = 4
    
    def __str__(self):
        return self.name