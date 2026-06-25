from typing import NamedTuple, Optional

class ImageLocation(NamedTuple):
    filename: str
    position_in_stack: Optional[int] = None
    
    @staticmethod
    def parse(location: str) -> 'ImageLocation':
        STACK_INDEXER = '@'
        parts = location.split(STACK_INDEXER, maxsplit=1)
        
        if len(parts) == 2:
            return ImageLocation(
                filename=parts[1], 
                position_in_stack=int(parts[0])
            )
        else:
            assert(len(parts) == 1)
            return ImageLocation(filename=parts[0])
