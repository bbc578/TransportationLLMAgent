# TransportationLLMAgent

城市交通运行分析与事件处置 LLM-Agent 系统。

本项目面向城市交通运行监测、拥堵研判、交通事件处置、运行日报生成与信控优化辅助等场景，构建一个基于大语言模型的交通行业智能体应用原型。系统通过工具调用获取结构化交通数据，通过知识库检索提供交通治理与运行分析依据，并通过工作流编排完成多步骤业务任务。

项目重点不在于让大模型直接替代交通管理决策，而是探索如何将 LLM、Tool Calling、RAG、Workflow、Trace 与 Eval 等能力结合到智慧交通业务流程中，使模型能够在可控边界内完成查询、分析、解释和报告生成。

## Features

* 城市交通运行指标查询
* 拥堵路段 Top-K 分析
* 路段基础信息查询
* 交通事件查询与处置辅助
* 交通预测结果查询
* 样例信号方案查询
* 交通知识库问答
* 拥堵原因诊断 Workflow
* 交通事件处置 Workflow
* 交通运行日报生成 Workflow
* 信控优化辅助 Workflow
* 工具调用 Trace 记录
* Agent 行为评测与结果统计

## System Design

系统采用分层 Agent 架构：

```text
User Query
  ↓
API / Streamlit
  ↓
Executor
  ↓
Safety Check
  ↓
Planner
  ↓
LLM Client
  ↓
ToolRegistry / RAG / Workflow
  ↓
Answer Builder
  ↓
Trace / Eval
```

各模块职责如下：

* `LLM Client`：统一封装模型调用接口，支持 Mock 与 OpenAI-compatible 模型服务。
* `Planner`：将用户自然语言问题转换为结构化 AgentPlan。
* `Executor`：负责执行 Planner 生成的工具调用或工作流计划。
* `ToolRegistry`：维护可调用工具白名单，并负责工具参数校验与执行分发。
* `Tools`：提供交通指标查询、事件查询、预测结果查询、信号方案查询等确定性业务能力。
* `RAG`：面向交通运行分析、拥堵诊断、信控原则、事件处置流程等知识文档进行检索。
* `Workflow`：将复杂业务流程固化为多步骤任务，例如拥堵诊断、事件处置、日报生成。
* `Trace`：记录用户问题、AgentPlan、工具调用、工具结果、RAG 证据和最终回答。
* `Eval`：评估工具选择、结构化输出、RAG 引用、安全拒答和工作流执行效果。

## Core Scenarios

### 1. Traffic Metrics Query

Example:

```text
R1 区域平均速度是多少？
```

The system can call traffic data tools to retrieve average speed, traffic flow, occupancy, congestion index and related indicators.

### 2. Congestion Ranking

Example:

```text
今天哪些路段最拥堵？
```

The system can rank road segments by congestion index and return structured results.

### 3. Congestion Diagnosis

Example:

```text
S001 为什么拥堵？
```

The workflow may combine road segment information, traffic metrics, traffic events and knowledge base evidence to generate a diagnosis.

### 4. Incident Response Assistance

Example:

```text
某路段发生事故后应该如何处置？
```

The system can retrieve traffic incident records and relevant response guidelines, then generate an operational suggestion.

### 5. Traffic Daily Report

Example:

```text
生成一份早高峰交通运行日报。
```

The system can combine traffic metrics, congestion ranking, incident records and report templates to produce a structured traffic operation report.

### 6. Signal Optimization Assistance

Example:

```text
这个路口排队严重，有什么信控优化建议？
```

The system can provide auxiliary suggestions based on sample traffic metrics, signal plan data and signal control principles. Such suggestions are for analysis only and should not be directly deployed to real traffic signal systems without simulation validation and human review.

## Tool Calling

The project treats external business capabilities as controlled tools. The LLM does not directly access databases or execute business logic. It only generates structured tool calls. Actual execution is handled by the backend through `Executor` and `ToolRegistry`.

A typical tool call flow is:

```text
User Query
  ↓
Planner generates AgentPlan
  ↓
Executor checks tool name and arguments
  ↓
ToolRegistry validates input schema
  ↓
Tool function runs
  ↓
Output schema is validated
  ↓
Answer Builder generates final response
```

This design reduces hallucination and prevents the model from inventing traffic facts.

## RAG

The RAG module is used for traffic-related knowledge retrieval. It supports knowledge base construction, chunking, retrieval and citation return.

Typical knowledge documents include:

* traffic operation report templates
* congestion diagnosis guidelines
* traffic incident response procedures
* signal control principles
* digital twin traffic platform notes

When the knowledge base does not contain relevant evidence, the system should explicitly report that no reliable source was found instead of fabricating an answer.

## Workflow

The project includes several workflow-style tasks:

* `congestion_diagnosis_workflow`
* `incident_response_workflow`
* `traffic_daily_report_workflow`
* `signal_optimization_assistant_workflow`
* `traffic_knowledge_qa_workflow`

Workflows are used for stable multi-step business processes. They combine tools, retrieval and final answer generation in a controlled execution path.

## Safety Boundary

The system is designed for traffic operation analysis and decision support. It does not directly control real-world infrastructure.

The following actions are outside the system boundary:

* directly changing traffic signal timing
* deleting or modifying real traffic event records
* issuing real control commands to road infrastructure
* replacing traffic management authorities or human reviewers
* accessing private personal information
* executing arbitrary system commands

Signal optimization results are only auxiliary analysis suggestions and require simulation validation and human review before real-world use.

## Evaluation

The project includes Agent evaluation cases for measuring system behavior, including:

* intent recognition accuracy
* tool selection accuracy
* structured output validity
* RAG citation rate
* refusal accuracy
* workflow success rate
* trace saving rate

Evaluation is used to check whether the Agent selects the correct tools, follows workflow constraints, provides evidence-based answers and refuses unsafe requests.

## Tech Stack

* Python
* FastAPI
* Streamlit
* Pydantic
* OpenAI-compatible API
* Qwen-compatible model service
* RAG
* Tool Calling
* Workflow orchestration
* Agent Trace
* Agent Evaluation

## Project Scope

This repository is a research and engineering prototype for traffic-domain LLM-Agent applications. The sample data and knowledge base are used for demonstration and reproducible development. The system is not connected to real traffic management infrastructure and should not be interpreted as a production traffic control platform.

## License

MIT
