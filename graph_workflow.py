from typing import TypedDict, Dict, Optional
from langgraph.graph import StateGraph, END
from dashboard_analyzer import DashboardAnalyzer
from timeseries_analyzer import TimeSeriesAnalyzer
from domain_specific_analyzer import DomainSpecificAnalyzer
from chat_agent import ChatAgent
from pathlib import Path

# Состояние графа
class AgentState(TypedDict):
    image_path: Optional[str]
    data_path: Optional[str]
    dash_features: Optional[Dict]
    ts_features: Optional[Dict]
    general_annotation: Optional[str]
    domain_annotation: Optional[str]
    final_annotation: Optional[str]
    user_query: Optional[str]
    chat_history: list
    response: Optional[str]

def create_graph():
    """Создает и настраивает граф задач для анализа дашборда и временного ряда."""
    dashboard_analyzer = DashboardAnalyzer()
    timeseries_analyzer = TimeSeriesAnalyzer()
    domain_specific_analyzer = DomainSpecificAnalyzer(default_domain="")
    chat_agent = ChatAgent()

    graph = StateGraph(AgentState)

    # Узел для анализа изображения дашборда
    def analyze_dashboard(state: AgentState) -> AgentState:
        if state["image_path"]:
            state["dash_features"] = dashboard_analyzer.analyze_dashboard(state["image_path"])
        return state

    # Узел для анализа временного ряда
    def analyze_timeseries(state: AgentState) -> AgentState:
        if state["data_path"]:
            df, message = timeseries_analyzer.read_data(Path(state["data_path"]))
            if df is None:
                state["ts_features"] = {"error": message}
            else:
                state["ts_features"] = timeseries_analyzer.analyze_time_series(df)
        return state

    # Узел для генерации общей аннотации
    def generate_annotation(state: AgentState) -> AgentState:
        if state["dash_features"] and state["ts_features"]:
            state["general_annotation"] = chat_agent.generate_general_annotation(
                state["ts_features"], state["dash_features"]
            )
        return state

    # Узел для адаптации аннотации к домену
    def adapt_annotation(state: AgentState) -> AgentState:
        if state["general_annotation"]:
            state["domain_annotation"] = domain_specific_analyzer.adapt_to_domain(
                state["general_annotation"],
                state["image_path"],
                state["data_path"]
            )
        return state

    # Узел для проверки аннотации
    def review_annotation(state: AgentState) -> AgentState:
        if state["domain_annotation"] and state["dash_features"] and state["ts_features"]:
            state["final_annotation"] = chat_agent.review_annotation(
                state["domain_annotation"], state["ts_features"], state["dash_features"]
            )
        return state

    # Узел для обработки пользовательского запроса
    def process_query(state: AgentState) -> AgentState:
        if state["user_query"]:
            state["response"] = chat_agent.process_user_query(
                state["user_query"],
                state["image_path"],
                state["data_path"],
                state["chat_history"]
            )
        return state

    # Добавляем узлы в граф
    graph.add_node("analyze_dashboard", analyze_dashboard)
    graph.add_node("analyze_timeseries", analyze_timeseries)
    graph.add_node("generate_annotation", generate_annotation)
    graph.add_node("adapt_annotation", adapt_annotation)
    graph.add_node("review_annotation", review_annotation)
    graph.add_node("process_query", process_query)

    # Определяем последовательность выполнения
    graph.set_entry_point("analyze_dashboard")
    graph.add_edge("analyze_dashboard", "analyze_timeseries")
    graph.add_edge("analyze_timeseries", "generate_annotation")
    graph.add_edge("generate_annotation", "adapt_annotation")
    graph.add_edge("adapt_annotation", "review_annotation")
    graph.add_edge("review_annotation", END)
    graph.add_conditional_edges(
        "process_query",
        lambda state: END if state["response"] else "process_query"
    )

    return graph.compile()