This is a repository containing all the experiments performed for the research paper "Table-First Guarded Reasoning: Outperfomring Larger MLLMs in Chart Question Answering" submitted for IEEE COAI 2026 and part of the thesis graduation project at the AUC.
There are three folders:
- Chart2Table Pipeline + GUI Demo: contains the final model that produced the highest accuracy results on chart qa along with a comparator demo application. You'll find a server python files that contains the pipelined model, a server python file that contains only the base qwen model for comparison, and the gui application that sends requests to these servers. Please follow the Readme in each of the server folders to see how you can set them up and use the GUI app to connect to them and test them.

- Text2Struct Pipeline: contains the alternative pipeline discussed in the paper that uses Text2Struct instead of Chart2Table.

- Additional Experiments: various other trained models and experiements that we tried while making our research.
- 
