import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math


def plot_scatter(
    df,
    x,
    y,
    x_name,
    y_name,
    y_min,
    y_max,
    title="",
    orientation="h",
    height=400,
    width=700,
    fig_update=False,
    text=False,
    color="#636EFA",
    highlight=False,
    list_highlight=[],
    color_highlight="red",
    brazil=False,):

    if text == True:
        fig = px.scatter(df, x=x, y=y, hover_data=["country"], text="country")
        fig.update_traces(textposition="top center")
    else:
        fig = px.scatter(
            df,
            x=x,
            y=y,
            hover_data=["country"],
            labels={
                x: x_name,
                y: y_name,
            },
        )
    if len(title) > 0:
        fig.update_layout(height=height, width=width, title_text=title)
    else:
        fig.update_layout(height=height, width=width)
    if fig_update == True:
        fig.update_yaxes(range=[y_min, y_max])
    fig.update_traces(marker=dict(color=color))
    if highlight == True:
        fig.add_traces(
            px.scatter(df[df["country"].isin(list_highlight)], x=x, y=y)
            .update_traces(marker_color=color_highlight)
            .data
        )
    if brazil == True:
        fig.add_traces(
            px.scatter(df[df["country"] == "Brazil"], x=x, y=y)
            .update_traces(marker_color="green", marker_symbol="star", marker_size=10)
            .data
        )
    # fig.show("png")
    fig.show()


def plot_bar(
    df, text, yaxis_range, title, height=500, width=700, orientation="v", y_visible=False, y_title=0.9, x_title=0.5,):

    fig = px.bar(
        df,
        color=df.index,
        orientation=orientation,
        height=height,
        width=width,
        text=text,
        title=title,
        color_discrete_map={
            "Some rights to same-sex couples": "#3EC1CD",
            "Same-sex marriage legal": "#00CC96",
            "Same-sex marriage not legally recognized": "#EF553B",
        },
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        yaxis_title=None,
        margin=dict(l=20, r=20, t=70, b=0),
        yaxis_range=yaxis_range,
    )
    fig.update_layout(title={"y": y_title, "x": x_title, "xanchor": "center", "yanchor": "top"})
    fig.update_layout({"plot_bgcolor": "rgba(0,0,0,0)", "paper_bgcolor": "rgba(0,0,0,0)"})

    fig.update_yaxes(visible=y_visible, showticklabels=False)
    # fig.write_image("img/teste.png")
    # fig.show("png")
    fig.show()


def make_multiple_plots(df, list_indexes, x, height=3000, width=950, plot_type="scatter"):

    if len(list_indexes) % 2 == 0:
        rows = len(list_indexes) / 2
    else:
        rows = math.ceil(len(list_indexes) / 2)
    rows = int(rows)
    fig = make_subplots(rows=rows, cols=2, start_cell="top-left", subplot_titles=list_indexes)
    fig.update_layout(
        height=height,
        width=width,
        title_text=x,
        showlegend=False,
    )

    row = 1
    col = 1
    for i in list_indexes:
        if plot_type == "scatter":
            fig.add_trace(go.Scatter(x=df[x], y=df[i], mode="markers"), row=row, col=col)
        else:
            df2 = df.groupby(x)[i].mean().sort_values(ascending=False)
            fig.add_trace(go.Bar(x=df2.index, y=df2), row=row, col=col)
        col += 1
        if col == 3:
            row += 1
            col = 1

    # fig.show("png")
    fig.show()


def get_missing(df):
    missing = df.isnull().sum()
    missing_percentage = df.isnull().sum() / df.isnull().count() * 100
    missing_percentage = round(missing_percentage, 1)
    missing_data = pd.concat([missing, missing_percentage], axis=1, keys=["Total", "%"])
    missing_data = missing_data[missing_data["Total"] > 0].sort_values(by=["%"], ascending=False)

    return missing_data


def get_dataframe():

    dict_country = {
        "Bolivia": ("Bolivia", "Bolivia (Plurinational State of)", "Bolivia`"),
        "Brunei": ("Brunei", "Brunei Darussalam"),
        "Congo": (
            "Congo",
            "Congo (Democratic Republic of the)",
            "DR Congo",
            "Democratic Republic of Congo",
            "Democratic Republic of the Congo",
            "Republic of the Congo",
        ),
        "Czech Republic": ("Czechia", "Czech Republic"),
        "Dominican Republic": ("Dominica", "Dominican Republic"),
        "Eswatini": ("Eswatini", "Eswatini (Kingdom of)", "Eswatini (Swaziland)"),
        "Faroe Islands": ("Faeroe Islands", "Faroe Islands"),
        "Fiji": ("Fiji", "FijiFiji"),
        "Gambia": ("Gambia", "Gambia, The", "The Gambia"),
        "Guinea-Bissau": ("Guinea", "Guinea-Bissau"),
        "French Guiana": ("French Guiana", "Guyana"),
        "Hong Kong": ("Hong Kong", "Hong Kong", "China (SAR)", "China (Sar)"),
        "India": ("India", "India h", "India H"),
        "Iran": ("Iran", "Iran (Islamic Republic of)"),
        "Israel": ("Israel", "Israel J"),
        "Ivory Coast": ("Ivory Coast", "Côte D'Ivoire"),
        "Kyrgyzstan": ("Kyrgyz Republic", "Kyrgyzstan"),
        "Laos": ("Lao People'S Democratic Republic", "Laos"),
        "Macau": ("Macao", "Macau"),
        "Micronesia": (
            "Micronesia",
            "Micronesia (Federated States of)",
            "Micronesia (country)",
            "Micronesia (Country)",
            "Micronesia (Federated States Of)",
        ),
        "Moldova": ("Moldova", "Moldova (Republic of)"),
        "Nepal": ("Nepal", "Nepal H"),
        "Palestine": ("Palestine", "Palestine, State of", "Palestine, State Of"),
        "Russia": ("Russia", "Russian Federation"),
        "Saint Martin": ("Saint Martin", "Saint Martin (French part)", "Sint Maarten", "Sint Maarten (Dutch part)"),
        "Syria": ("Syria", "Syrian Arab Republic"),
        "Tanzania": ("Tanzania", "Tanzania (United Republic of)"),
        "Timor-Leste": ("Timor", "Timor-Leste"),
        "Venezuela": ("Venezuela", "Venezuela (Bolivarian Republic of)"),
        "Vietnam": ("Viet Nam", "Vietnam"),
        "South Korea": ("Korea (Republic Of)", "South Korea", "Korea, Rep."),
    }

    reversed_dict = {val: key for key in dict_country for val in dict_country[key]}

    df_corruption = pd.read_csv(r"../data/corruptionScore.csv")
    df_corruption = df_corruption.drop("pop2021", axis=1)
    df_corruption = df_corruption.rename(columns={"corruptionScore": "corruption_score"})
    df_corruption["country"] = df_corruption["country"].map(reversed_dict).fillna(df_corruption["country"])

    df_free_expression = pd.read_csv(r"../data/free_expression.csv")
    df_free_expression = df_free_expression.drop("pop2021", axis=1)
    df_free_expression = df_free_expression.rename(columns={"freeExpressionIndex": "free_expression_index"})
    df_free_expression["country"] = (
        df_free_expression["country"].map(reversed_dict).fillna(df_free_expression["country"])
    )

    df_gdppc = pd.read_csv(r"../data/gdppc.csv")
    df_gdppc = df_gdppc.drop("pop2021", axis=1)
    df_gdppc = df_gdppc.rename(columns={"ranking": "gdppc_ranking"})
    df_gdppc["country"] = df_gdppc["country"].map(reversed_dict).fillna(df_gdppc["country"])

    df_happiness = pd.read_csv(r"../data/happiness.csv")
    df_happiness = df_happiness.drop("pop2021", axis=1)
    df_happiness = df_happiness.rename(
        columns={
            "rank": "happiness_ranking",
            "happiness2021": "happiness_2021",
            "happiness2020": "happiness_2020",
        }
    )
    df_happiness["country"] = df_happiness["country"].map(reversed_dict).fillna(df_happiness["country"])

    df_happiness_score = pd.read_csv(r"../data/happinessRankScore.csv")
    df_happiness_score = df_happiness_score.drop("pop2021", axis=1)
    df_happiness_score = df_happiness_score.rename(
        columns={"happinessRank": "happiness_ranking2", "happinessScore": "happiness_score"}
    )
    df_happiness_score["country"] = (
        df_happiness_score["country"].map(reversed_dict).fillna(df_happiness_score["country"])
    )

    df_hdi = pd.read_csv(r"../data/hdi.csv")
    df_hdi = df_hdi.drop("pop2021", axis=1)
    df_hdi = df_hdi.rename(columns={"ranking": "hdi_ranking_2021"})
    df_hdi["country"] = df_hdi["country"].map(reversed_dict).fillna(df_hdi["country"])

    df_hdi3 = pd.read_csv(r"../data/hdi2019.csv")
    df_hdi3 = df_hdi3.drop("pop2021", axis=1)
    df_hdi3 = df_hdi3.rename(columns={"hdi2019": "hdi_2019"})
    df_hdi3["country"] = df_hdi3["country"].map(reversed_dict).fillna(df_hdi3["country"])

    df_human_rights_scores = pd.read_csv(r"../data/human_rights_scores.csv")
    df_human_rights_scores = df_human_rights_scores[df_human_rights_scores["Year"] == 2017]
    df_human_rights_scores = df_human_rights_scores.rename(
        columns={
            "Entity": "country",
            "Human Rights Score (Schnakenberg & Fariss, 2014; Fariss, 2019)": "human_rights_score_2019",
        }
    )
    df_human_rights_scores = df_human_rights_scores.drop(["Code", "Year"], axis=1)
    df_human_rights_scores["country"] = (
        df_human_rights_scores["country"].map(reversed_dict).fillna(df_human_rights_scores["country"])
    )

    df_iq = pd.read_csv(r"../data/iq.csv")
    df_iq = df_iq.drop("pop2021", axis=1)
    df_iq["country"] = df_iq["country"].map(reversed_dict).fillna(df_iq["country"])

    df_legalize = pd.read_csv(r"../data/legalize.csv")
    df_legalize = df_legalize.drop("pop2021", axis=1)
    df_legalize = df_legalize.rename(columns={"legalizeYear": "legalize_year"})
    df_legalize["country"] = df_legalize["country"].map(reversed_dict).fillna(df_legalize["country"])

    df_peace = pd.read_csv(r"../data/peace.csv")
    df_peace = df_peace.drop("pop2021", axis=1)
    df_peace = df_peace.rename(columns={"peaceIndex": "peace_index", "rank": "peace_index_ranking"})
    df_peace["country"] = df_peace["country"].map(reversed_dict).fillna(df_peace["country"])

    df_population = pd.read_csv(r"../data/pop_2021.csv")
    df_population = df_population.rename(columns={"pop2021": "population_2021"})
    df_population["country"] = df_population["country"].map(reversed_dict).fillna(df_population["country"])

    df_racism = pd.read_csv(r"../data/racism.csv")
    df_racism = df_racism.drop("pop2021", axis=1)
    df_racism = df_racism.rename(columns={"ranking": "racism_ranking"})
    df_racism["country"] = df_racism["country"].map(reversed_dict).fillna(df_racism["country"])

    df_same_sex_marriage = pd.read_csv(r"../data/same_sex_marriage_recognition.csv")
    df_same_sex_marriage = df_same_sex_marriage.rename(
        columns={
            "Entity": "country",
            "Year": "same_sex_marriage_year",
            "Same sex marriage and civil unions legal": "same_sex_marriage",
        }
    )
    df_same_sex_marriage = df_same_sex_marriage.drop("Code", axis=1)
    df_same_sex_marriage["country"] = (
        df_same_sex_marriage["country"].map(reversed_dict).fillna(df_same_sex_marriage["country"])
    )

    df_score = pd.read_csv(r"../data/score.csv")
    df_score = df_score.drop("pop2021", axis=1)
    df_score = df_score.rename(columns={"ranking": "ranking_score"})
    df_score["country"] = df_score["country"].map(reversed_dict).fillna(df_score["country"])

    df_wealth_poverty = pd.read_csv(r"../data/wealth_poverty.csv")
    df_wealth_poverty = df_wealth_poverty.rename(
        columns={
            "netGini": "net_gini",
            "wealthGini": "wealth_gini",
            "medianIncome": "median_income",
            "povertyRate": "poverty_rate",
        }
    )
    df_wealth_poverty["country"] = df_wealth_poverty["country"].map(reversed_dict).fillna(df_wealth_poverty["country"])

    df_women = pd.read_csv(r"../data/women_cant_vote.csv")
    df_women = df_women.drop("pop2021", axis=1)
    df_women["women_cant_vote"] = 1
    df_women["country"] = df_women["country"].map(reversed_dict).fillna(df_women["country"])

    df_gender_inequality = pd.read_excel(r"../data/gender_inequality.xlsx")
    df_gender_inequality = df_gender_inequality.rename(
        columns={
            "HDI rank": "hdi_rank",
            "Country": "country",
            "Gender Inequality Index 2019 Value": "gender_inequality_index_2019",
            "Gender Inequality Index 2019 Rank": "gender_inequality_index_2019_ranking",
            "Share of seats in parliament (% held by women) 2019": "share_seats_parliament_women_2019",
            "Population with at least some secondary education (% ages 25 and older) Female 2015–2019": "population_secondary_education_ages_25_older_female",
            "Labour force participation rate  (% ages 15 and older) Female 2019": "labour_force_participation_rate_ages_15_older_female_2019",
        }
    )
    df_gender_inequality = df_gender_inequality.drop("hdi_rank", axis=1)
    df_gender_inequality["country"] = (
        df_gender_inequality["country"].map(reversed_dict).fillna(df_gender_inequality["country"])
    )

    df_gender_inequality2 = pd.read_excel(r"../data/gender_inequality_2.xlsx")
    df_gender_inequality2 = df_gender_inequality2.rename(
        columns={
            "HDI rank": "hdi_rank",
            "Country": "country",
            "Population with at least some secondary education 2015–2019 (female to male ratio)": "population_some_secondary_education_2019_female_to_male_ratio",
            "Total unemployment rate (female to male ratio) 2019": "total_unemployment_rate_female_male_ratio_2019",
            "Share of seats held by women In parliament 2019": "share_seats_held_women_parliament_2019",
            "Share of seats held by women In local government 2017–2019": "share_seats_held_women_local_government_2019",
        }
    )
    df_gender_inequality2 = df_gender_inequality2.drop("hdi_rank", axis=1)
    df_gender_inequality2["country"] = (
        df_gender_inequality2["country"].map(reversed_dict).fillna(df_gender_inequality2["country"])
    )

    df_hdi2 = pd.read_excel(r"../data/hdi_by_year.xlsx")
    df_hdi2 = df_hdi2.rename(columns={"Country": "country", "2021[17]": "hdi_2021"})
    df_hdi2 = df_hdi2[["country", "hdi_2021"]]
    df_hdi2["country"] = df_hdi2["country"].map(reversed_dict).fillna(df_hdi2["country"])

    df_indexes1 = pd.read_excel(r"../data/indexes1.xlsx")
    df_indexes1 = df_indexes1.rename(
        columns={
            "HAPPINESS": "happiness",
            " Per capita GDP ": "per_capita_gdp",
            "% with tertiary educ.": "percent_tertiary_education",
            "GRI": "gri",
            "SHI": "shi",
            "legality of sam-sex sexual-activity": "legality_sam_sex_sexual_activity",
            "marriage/ civil unions": "marriage_civil_unions",
            "same-sex coupleadoptions": "same_sex_coupleadoptions",
            "antidiscrimination laws": "antidiscrimination_laws",
            "Democracy Status": "democracy_status",
            "civil liberties": "civil_liberties",
            "Healthy life expectancy": "healthy_life_expectancy",
            "Freedom to make life choices": "freedom_life_choices",
            "Perceptions of corruption": "perceptions_corruption",
        }
    )
    df_indexes1 = df_indexes1.drop(
        [
            "sub contient",
            "Predom. Religion",
            "DEMOC.",
            "Religiosity",
            "SOGI-LI",
            "serving in the military",
            " gender ID markers",
            "electoral process and pluralism",
            "funct. Of gov",
            "polit. Particpation",
            "political culture",
            "Satisfaction with GDP per capita",
            "Social support",
            "Generosity",
        ],
        axis=1,
    )
    df_indexes1["country"] = df_indexes1["country"].map(reversed_dict).fillna(df_indexes1["country"])

    df_inequality = pd.read_excel(r"../data/inequality.xlsx")
    df_inequality = df_inequality.rename(
        columns={
            "Country": "country",
            "Coefficient of human inequality-2019": "coefficient_human_inequality_2019",
            "Inequality in education-percent-2019": "inequality_education_percent_2019",
            "Inequality-adjusted education index-2019": "inequality_adjusted_education_index_2019",
            "Inequality in income-percent-2019": "inequality_in_income_percent_2019",
            "Inequality-adjusted income index=2019": "inequality_adjusted_income_index_2019",
            "Gini coefficient-2010-2018": "gini_coefficient_2010_2018",
        }
    )
    df_inequality = df_inequality.drop(
        ["HDI rank", " Richest 1 percent-2010-2017", "Richest 10 percent-2010-2018", "Poorest 40 percent-2010-2018"],
        axis=1,
    )
    df_inequality["country"] = df_inequality["country"].map(reversed_dict).fillna(df_inequality["country"])

    df_literacy = pd.read_excel(r"../data/literacy.xlsx")
    df_literacy = df_literacy.rename(
        columns={
            "Country": "country",
            "Literacy rate Adult                                 (% ages 15 and older) 2008-2018": "literacy_rate_adult_percent_ages_15_and_older_2008_2018",
            "Population with at least some secondary education (% ages 25 and older) 2015-2019": "population_with_some_secondary_education_perncent_25_and_older_2015_2019",
            "Government expenditure on education  (% of GDP) 2013-2018": "government_expenditure_education_percent_GDP_2013_2018",
        }
    )
    df_literacy = df_literacy.drop("HDI rank", axis=1)
    df_literacy["country"] = df_literacy["country"].map(reversed_dict).fillna(df_literacy["country"])

    df_per_capita = pd.read_excel(r"../data/per_capita.xlsx")
    df_per_capita = df_per_capita.rename(
        columns={
            "HDI rank": "hdi_rank",
            "Country": "country",
            "Gross domestic product (GDP) (2017 PPP $ billions) 2019": "gdp_2019",
            "Per capita (2017 PPP $) 2019": "per_capita_2019",
        }
    )
    df_per_capita["country"] = df_per_capita["country"].map(reversed_dict).fillna(df_per_capita["country"])

    df_poverty = pd.read_excel(r"../data/poverty.xlsx")
    df_poverty = df_poverty.rename(
        columns={
            "Country": "country",
            "Multidimensional Poverty Index index Value": "multidimensional_poverty_index_value",
            "Headcount %": "headcount_percent",
            "Intensity of deprivation %": "intensity_of_deprivation_percent",
            "Number of poor (year of the survey) (thousands)": "number_of_poor_thousands",
            "Number of poor (2018) (thousands)": "number_of_poor_2018_thousands",
            "Inequality among the poor Value": "inequality_among_the_poor_value",
            "Population in severe multidimensional poverty  (%)": "population_in_severe_multidimensional_poverty_percent",
            "Population living below income poverty line National poverty line %": "population_living_below_income_poverty_line_national_poverty_line_percent",
        }
    )
    df_poverty = df_poverty.drop(
        [
            "Population vulnerable to multidimensional poverty %",
            "Contribution of deprivation in dimension to overall multidimensional poverty Health",
            "Contribution of deprivation in dimension to overall multidimensional poverty Education %",
            "Contribution of deprivation in dimension to overall multidimensional poverty Standard of living %",
            "Population vulnerable to multidimensional poverty %",
            "Contribution of deprivation in dimension to overall multidimensional poverty Health",
            "Contribution of deprivation in dimension to overall multidimensional poverty Education %",
            "Contribution of deprivation in dimension to overall multidimensional poverty Standard of living %",
            "Population living below income poverty line PPP $1.90 a day %",
        ],
        axis=1,
    )
    df_poverty["country"] = df_poverty["country"].map(reversed_dict).fillna(df_poverty["country"])

    df_schooling = pd.read_excel(r"../data/schooling.xlsx")
    df_schooling = df_schooling.rename(
        columns={"Country": "country", "Gender Development Index-2019": "gender_development_index_2019"}
    )
    df_schooling = df_schooling[["country", "gender_development_index_2019"]]
    df_schooling["country"] = df_schooling["country"].map(reversed_dict).fillna(df_schooling["country"])

    df_indexes = pd.read_excel(r"../data/tb_indexes.xlsx")
    df_indexes = df_indexes.rename(
        columns={
            "HDI rank": "hdi_rank",
            "Country": "country",
            "Human Development Index (HDI) -value": "hdi_value",
            "Life expectancy at birth - years": "life_expectancy_at_birth_years",
            "Expected years of schooling - years": "expected_years_schooling_years",
            "Mean years of schooling - years": "mean_years_schooling_years",
            "Gross national income (GNI) per capita - 2017 PPP $": "gross_national_income_per_capita_2017",
        }
    )
    df_indexes = df_indexes.drop(["GNI per capita rank minus HDI rank - 2019", "HDI rank - 2018"], axis=1)
    df_indexes = df_indexes.drop("hdi_rank", axis=1)
    df_indexes["country"] = df_indexes["country"].map(reversed_dict).fillna(df_indexes["country"])

    df_population = df_population[
        ~df_population["country"].isin(
            ["Congo", "Guinea-Bissau", "Dominican Republic", "French Guiana", "Saint Martin"]
        )
    ]

    duplicated_countries = ["Congo", "Guinea-Bissau", "Dominican Republic", "French Guiana", "Saint Martin"]

    df_population = df_population[~df_population["country"].isin(duplicated_countries)]
    df = df_population.merge(df_corruption, on="country", how="left")
    df = df.merge(df_free_expression, on="country", how="left")
    df = df.merge(df_gdppc, on="country", how="left")
    df = df.merge(df_happiness, on="country", how="left")
    df = df.merge(df_happiness_score, on="country", how="left")
    df = df.merge(df_hdi, on="country", how="left")
    df = df.merge(df_hdi3, on="country", how="left")
    df = df.merge(df_human_rights_scores, on="country", how="left")
    df = df.merge(df_iq, on="country", how="left")
    df = df.merge(df_legalize, on="country", how="left")
    df = df.merge(df_peace, on="country", how="left")
    df = df.merge(df_racism, on="country", how="left")
    df = df.merge(df_same_sex_marriage, on="country", how="left")
    df = df.merge(df_score, on="country", how="left")
    df = df.merge(df_wealth_poverty, on="country", how="left")
    df = df.merge(df_women, on="country", how="left")
    df = df.merge(df_gender_inequality, on="country", how="left")
    df = df.merge(df_gender_inequality2, on="country", how="left")
    df = df.merge(df_hdi2, on="country", how="left")
    df = df.merge(df_indexes1, on="country", how="left")
    df = df.merge(df_inequality, on="country", how="left")
    df = df.merge(df_literacy, on="country", how="left")
    df = df.merge(df_per_capita, on="country", how="left")
    df = df.merge(df_poverty, on="country", how="left")
    df = df.merge(df_schooling, on="country", how="left")
    df = df.merge(df_indexes, on="country", how="left")

    cols = [
        "gender_development_index_2019",
        "gender_inequality_index_2019",
        "gender_inequality_index_2019_ranking",
        "share_seats_held_women_local_government_2019",
        "share_seats_held_women_parliament_2019",
        "total_unemployment_rate_female_male_ratio_2019",
        "literacy_rate_adult_percent_ages_15_and_older_2008_2018",
        "population_with_some_secondary_education_perncent_25_and_older_2015_2019",
        "government_expenditure_education_percent_GDP_2013_2018",
        "coefficient_human_inequality_2019",
        "per_capita_2019",
        "gdp_2019",
        "gini_coefficient_2010_2018",
    ]

    for col in cols:
        df[col] = df[col].replace("..", np.nan).astype(float)

    columns_to_keep = [
        "country",
        "hdi",
        "hdi_ranking_2021",
        "gdppc",
        "gdppc_ranking",
        "poverty_rate",
        "median_income",
        "net_gini",
        "gini_coefficient_2010_2018",
        "corruption_score",
        "happiness_2020",
        "happiness",
        "happiness_ranking",
        "happiness_ranking2",
        "happiness_score",
        "same_sex_marriage",
        "gender_development_index_2019",
        "gender_inequality_index_2019",
        "gender_inequality_index_2019_ranking",
        "share_seats_held_women_local_government_2019",
        "share_seats_held_women_parliament_2019",
        "total_unemployment_rate_female_male_ratio_2019",
        "literacy_rate_adult_percent_ages_15_and_older_2008_2018",
        "population_with_some_secondary_education_perncent_25_and_older_2015_2019",
        "expected_years_schooling_years",
        "mean_years_schooling_years",
        "government_expenditure_education_percent_GDP_2013_2018",
        "racism_ranking",
        "coefficient_human_inequality_2019",
        "human_rights_score_2019",
        "healthy_life_expectancy",
        "iq",
        "civil_liberties",
        "gri",
        "shi",
    ]

    for country in ["Switzerland", "Chile", "Slovenia"]:
        df["same_sex_marriage"].mask(df["country"] == country, "Same-sex marriage legal", inplace=True)

    df = df[columns_to_keep]
    df = df.sort_values("country")

    return df
