# Hill-Climbing-and-Simulated-Annealing-AI-Algorithms
Python version

https://www.udemy.com/course/hill-climbing-and-simulated-annealing-ai-algorithms/

這是一門介紹爬山演算法HC/模擬退火SA的課程

1. HC/SA 是屬於一種概念型的最佳化演算法，因此X沒有特定的更新方式，故可以用來解決各種問題

2. HC/SA 不屬於群體智能算法，因此P固定為1，因此計算成本極低但也表示求解精度可能不足

3. HC適合求解無局最優的簡單問題，例如Sphere；SA適合求解有局部最優的複雜問題

4. HC在每次迭代時，都會產生多個臨時的候選解(數量由設計者決定)，演算法會從中挑選一個最好的候選解作為下一代，若下一代支表現沒有優於當前，則終止

5. SA在每次迭代只會產生一個解作為下一代，若下一代支表現沒有優於當前，則一定機率接受並繼續迭代，或者終止

------------------------------------------------------------------------------------------------

Lesson 06:求最大化問題的HC，裡面有四種測試函數可以玩。因為跑了3萬代都沒結束計算，所以我加入了提早終止條件。

Lesson 10:SA，還沒看

Lesson 11:用HC/SA求解旅行銷售員問題TSP，我沒寫
