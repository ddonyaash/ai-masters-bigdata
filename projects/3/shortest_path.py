import sys
from pyspark import SparkContext, SparkConf

def main():
    start_node = sys.argv[1]
    end_node = sys.argv[2]
    dataset_path = sys.argv[3]
    output_path = sys.argv[4]

    conf = SparkConf().setAppName("BFS_ShortestPath")
    sc = SparkContext(conf=conf)

    edges = sc.textFile(dataset_path) \
        .map(lambda line: line.split('\t')) \
        .map(lambda x: (x[1], x[0])) \
        .groupByKey() \
        .mapValues(list) \
        .cache()

    paths = sc.parallelize([(start_node, (start_node,))])

    while True:
        found = paths.filter(lambda x: x[0] == end_node)
        
        if not found.isEmpty():
            result = found.map(lambda x: ",".join(x[1]))
            result.saveAsTextFile(output_path)
            break

        paths = paths.join(edges) \
            .flatMap(lambda x: [(neighbor, x[1][0] + (neighbor,)) 
                               for neighbor in x[1][1] 
                               if neighbor not in x[1][0]]) 
        
        if paths.isEmpty():
            break

    sc.stop()

if __name__ == "__main__":
    main()
