from .DropParser import DropParser
from .DatabricksDollyParser import DatabricksDollyParser
from .CovidQAParser import CovidQAParser
from .DatasetParser import DatasetParser

BENCHMARKS = [
    DropParser, 
    DatabricksDollyParser, 
    CovidQAParser
]

def get_benchmarks_display_names():
    return [{"id": benchmark.id, "display_name": benchmark.display_name} for benchmark in BENCHMARKS]

def get_benchmark(id):
    for benchmark in BENCHMARKS:
        if benchmark.id == id:
            return benchmark
