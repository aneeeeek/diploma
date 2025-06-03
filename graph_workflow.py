from typing import TypedDict, Dict, Optional
from langgraph.graph import StateGraph, END
from dashboard_analyzer import DashboardAnalyzer
from timeseries_analyzer import TimeSeriesAnalyzer
from domain_specific_analyzer import DomainSpecificAnalyzer
from chat_agent import ChatAgent
from pathlib import Path


class AgentState(TypedDict):
    image_path: Optional[str]
    data_path: Optional[str]
    dash_features: Optional[Dict]
    domain_features: Optional[Dict]
    ts_features: Optional[Dict]
    final_annotation: Optional[str]
    user_query: Optional[str]
    chat_history: list
    response: Optional[str]


def create_graph():
    """Создает и настраивает граф задач для анализа дашборда и временного ряда."""
    dashboard_analyzer = DashboardAnalyzer()
    timeseries_analyzer = TimeSeriesAnalyzer()
    domain_specific_analyzer = DomainSpecificAnalyzer(default_domain="finance")
    chat_agent = ChatAgent()

    graph = StateGraph(AgentState)

    def analyze_dashboard(state: AgentState) -> AgentState:
        if state["image_path"]:
            state["dash_features"] = dashboard_analyzer.analyze_dashboard(state["image_path"])
        return state

    def analyze_domain(state: AgentState) -> AgentState:
        if state["image_path"] or state["data_path"]:
            state["domain_features"] = domain_specific_analyzer.suggest_domain(
                state["image_path"], state["data_path"]
            )
        return state

    def analyze_timeseries(state: AgentState) -> AgentState:
        if state["data_path"]:
            df, message = timeseries_analyzer.read_data(Path(state["data_path"]))
            if df is None:
                state["ts_features"] = {
                    "metric": state["dash_features"].get("main_metric", "неизвестно") if state["dash_features"] else "неизвестно",
                    "domain": state["domain_features"].get("domain", "finance") if state["domain_features"] else "finance",
                    "trend": "неизвестно",
                    "seasonality": "неизвестно",
                    "min_value": "неизвестно",
                    "max_value": "неизвестно",
                    "anomalies": [],
                    "hypotheses": message
                }
            else:
                state["ts_features"] = timeseries_analyzer.analyze_time_series(
                    df,
                    state["image_path"],
                    state["dash_features"].get("main_metric", "неизвестно") if state["dash_features"] else "неизвестно",
                    state["domain_features"].get("domain", "finance") if state["domain_features"] else "finance"
                )
        return state

    def generate_annotation(state: AgentState) -> AgentState:
        if state["ts_features"] and not state["user_query"]:
            state["final_annotation"] = chat_agent.generate_general_annotation(state["ts_features"])
        return state

    def process_query(state: AgentState) -> AgentState:
        if state["user_query"]:
            state["response"] = chat_agent.process_user_query(
                state["user_query"],
                state["image_path"],
                state["data_path"],
                state["chat_history"],
                state["dash_features"],
                state["domain_features"],
                state["ts_features"]
            )
        return state

    graph.add_node("analyze_dashboard", analyze_dashboard)
    graph.add_node("analyze_domain", analyze_domain)
    graph.add_node("analyze_timeseries", analyze_timeseries)
    graph.add_node("generate_annotation", generate_annotation)
    graph.add_node("process_query", process_query)

    graph.set_entry_point("analyze_dashboard")
    graph.add_edge("analyze_dashboard", "analyze_domain")
    graph.add_edge("analyze_domain", "analyze_timeseries")
    graph.add_edge("analyze_timeseries", "generate_annotation")
    graph.add_edge("generate_annotation", "process_query")
    graph.add_edge("process_query", END)

    return graph.compile()