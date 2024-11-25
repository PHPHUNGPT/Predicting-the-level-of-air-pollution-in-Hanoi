import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Đọc dữ liệu
cleaned_data = pd.read_csv("data/cleaned_air_quality_data.csv")
evaluation_results = pd.read_csv("resources/model_evaluation_results.csv")

# Tạo danh sách đặc trưng
feature_list = cleaned_data.columns.tolist()

# Tạo Dashboard
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Dashboard: Data & Model Evaluation"

# Tab 1: Dữ Liệu Sạch
tab1_content = html.Div([
    html.H3("Dữ Liệu Sạch", style={"textAlign": "center"}),

    # Thanh tùy chỉnh
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Chọn loại biểu đồ:", style={"margin-right": "10px"}),
                dcc.Dropdown(
                    id="chart-type",
                    options=[
                        {"label": "Biểu đồ đường", "value": "line"},
                        {"label": "Biểu đồ cột", "value": "bar"},
                        {"label": "Biểu đồ phân tán", "value": "scatter"},
                        {"label": "Box-Plot", "value": "box"},
                        {"label": "Biểu đồ tròn (Pie)", "value": "pie"}
                    ],
                    value="line",
                    style={"width": "100%"}
                )
            ], width=6),
            dbc.Col([
                html.Label("Chọn biến cần vẽ:", style={"margin-right": "10px"}),
                dcc.Dropdown(
                    id="selected-column",
                    options=[{"label": col, "value": col} for col in feature_list],
                    value=feature_list[0],
                    style={"width": "100%"}
                )
            ], width=6),
        ], align="center", className="mb-4"),
    ]),

    # Biểu đồ tùy chỉnh
    dcc.Graph(id="dynamic-chart"),

    # Ma trận tương quan
    dcc.Graph(
        id="correlation-matrix",
        figure=px.imshow(cleaned_data.corr(), text_auto=True, color_continuous_scale="Viridis",
                         title="Ma trận tương quan").update_layout(template="plotly_white")
    ),

    # Bảng thống kê mô tả dữ liệu
    dash_table.DataTable(
        id="data-table",
        columns=[{"name": col, "id": col} for col in cleaned_data.describe().transpose().reset_index().columns],
        data=cleaned_data.describe().transpose().reset_index().to_dict("records"),
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "rgb(230, 230, 230)", "color": "black"},
        style_data={"backgroundColor": "rgb(250, 250, 250)", "color": "black"},
    )
])

# Tab 2: Kết Quả Đánh Giá Mô Hình
tab2_content = html.Div([
    html.H3("Kết Quả Đánh Giá Mô Hình", style={"textAlign": "center"}),

    # Biểu đồ so sánh MAE, MSE, R²
    html.Div([
        dcc.Graph(id="mae-comparison"),
        dcc.Graph(id="mse-comparison"),
        dcc.Graph(id="r2-comparison"),
    ]),

    # Bảng kết quả đánh giá
    dash_table.DataTable(
        id="evaluation-table",
        columns=[
            {"name": "Model", "id": "Model"},
            {"name": "MAE", "id": "MAE"},
            {"name": "MSE", "id": "MSE"},
            {"name": "R²", "id": "R²"}
        ],
        data=evaluation_results.to_dict("records"),
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "rgb(230, 230, 230)", "color": "black"},
        style_data={"backgroundColor": "rgb(250, 250, 250)", "color": "black"},
    )
])

# Layout chính của Dashboard
app.layout = dbc.Container([
    dbc.Tabs([
        dbc.Tab(tab1_content, label="Dữ Liệu Sạch"),
        dbc.Tab(tab2_content, label="Kết Quả Đánh Giá Mô Hình")
    ])
], fluid=True)

# Callback cho Tab 1: Dữ Liệu Sạch
@app.callback(
    Output("dynamic-chart", "figure"),
    [Input("chart-type", "value"),
     Input("selected-column", "value")]
)
def update_chart(chart_type, selected_column):
    if chart_type == "line":
        fig = px.line(cleaned_data, y=selected_column, title=f"Biểu đồ đường: {selected_column}")
    elif chart_type == "bar":
        fig = px.bar(cleaned_data, y=selected_column, title=f"Biểu đồ cột: {selected_column}")
    elif chart_type == "scatter":
        fig = px.scatter(cleaned_data, y=selected_column, title=f"Biểu đồ phân tán: {selected_column}")
    elif chart_type == "box":
        fig = px.box(cleaned_data, y=selected_column, title=f"Box-Plot: {selected_column}")
    elif chart_type == "pie":
        fig = px.pie(cleaned_data, names=selected_column, title=f"Biểu đồ tròn: {selected_column}")
    else:
        fig = go.Figure()
    return fig.update_layout(template="plotly_white")

# Callback cho Tab 2: Biểu đồ So Sánh Chỉ Số
@app.callback(
    [Output("mae-comparison", "figure"),
     Output("mse-comparison", "figure"),
     Output("r2-comparison", "figure")],
    Input("evaluation-table", "data")
)
def update_model_comparisons(data):
    if not data:
        return go.Figure(), go.Figure(), go.Figure()

    df = pd.DataFrame(data)
    mae_fig = px.bar(df, x="Model", y="MAE", title="So sánh MAE giữa các mô hình")
    mse_fig = px.bar(df, x="Model", y="MSE", title="So sánh MSE giữa các mô hình")
    r2_fig = px.bar(df, x="Model", y="R^2", title="So sánh R² giữa các mô hình")

    return mae_fig, mse_fig, r2_fig

# Khởi chạy ứng dụng
if __name__ == "__main__":
    app.run_server(debug=True)
