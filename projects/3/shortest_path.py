import sys
from pyspark import SparkContext, SparkConf

def main():
    # Считываем аргументы
    start_node = sys.argv[1]
    end_node = sys.argv[2]
    dataset_path = sys.argv[3]
    output_path = sys.argv[4]

    conf = SparkConf().setAppName("BFS_ShortestPath")
    sc = SparkContext(conf=conf)

    # 1. Загрузка графа. Инвертируем по условию: (follower_id, user_id) -> (src, dst)
    # Используем groupByKey, чтобы для каждого узла один раз знать всех его соседей
    edges = sc.textFile(dataset_path) \
        .map(lambda line: line.split('\t')) \
        .map(lambda x: (x[1], x[0])) \
        .groupByKey() \
        .mapValues(list) \
        .cache()

    # 2. Инициализация путей: (текущий_узел, путь_к_нему)
    # Путь храним как кортеж строк, чтобы легче было джойнить
    paths = sc.parallelize([(start_node, (start_node,))])

    while True:
        # Проверяем, достигли ли мы цели
        found = paths.filter(lambda x: x[0] == end_node)
        
        if not found.isEmpty():
            # Если нашли, берем только сами пути, сортируем для чекера и форматируем
            # Оставляем только пути минимальной длины (они уже тут, так как это BFS)
            result = found.map(lambda x: ",".join(x[1]))
            result.saveAsTextFile(output_path)
            break

        # Делаем шаг BFS: Join с графом
        # x[0] - текущая вершина, x[1][0] - путь до неё, x[1][1] - соседи
        paths = paths.join(edges) \
            .flatMap(lambda x: [(neighbor, x[1][0] + (neighbor,)) 
                               for neighbor in x[1][1] 
                               if neighbor not in x[1][0]]) # защита от циклов
        
        # Если новых путей нет, а цель не найдена - выходим
        if paths.isEmpty():
            break

    sc.stop()

if __name__ == "__main__":
    main()
