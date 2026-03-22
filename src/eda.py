import seaborn as sns
import matplotlib.pyplot as plt

def run_eda(df,q):

    fig,ax=plt.subplots(figsize=(8,5))

    plots={
1:("Price Distribution",lambda:sns.histplot(df["Price_in_Lakhs"],kde=True,ax=ax)),
2:("Size Distribution",lambda:sns.histplot(df["Size_in_SqFt"],kde=True,ax=ax)),
3:("Price/SqFt by Property Type",lambda:sns.boxplot(x="Property_Type",y="Price_per_SqFt",data=df,ax=ax)),
4:("Size vs Price",lambda:sns.scatterplot(x="Size_in_SqFt",y="Price_in_Lakhs",data=df,ax=ax)),
5:("Outliers",lambda:sns.boxplot(y=df["Price_per_SqFt"],ax=ax)),
6:("State Price",lambda:df.groupby("State")["Price_per_SqFt"].mean().plot(kind="bar",ax=ax)),
7:("City Price",lambda:df.groupby("City")["Price_in_Lakhs"].mean().plot(kind="bar",ax=ax)),
8:("Age Locality",lambda:df.groupby("Locality")["Property_Age"].median().head(10).plot(kind="bar",ax=ax)),
9:("BHK Count",lambda:sns.countplot(x="BHK",data=df,ax=ax)),
10:("Top Localities",lambda:df.groupby("Locality")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(5).plot(kind="bar",ax=ax)),
11:("Correlation",lambda:sns.heatmap(df.corr(numeric_only=True),cmap="coolwarm",ax=ax)),
12:("Schools Impact",lambda:sns.scatterplot(x="Nearby_Schools",y="Price_per_SqFt",data=df,ax=ax)),
13:("Hospital Impact",lambda:sns.scatterplot(x="Nearby_Hospitals",y="Price_per_SqFt",data=df,ax=ax)),
14:("Furnished Price",lambda:sns.boxplot(x="Furnished_Status",y="Price_in_Lakhs",data=df,ax=ax)),
15:("Facing Impact",lambda:sns.boxplot(x="Facing",y="Price_per_SqFt",data=df,ax=ax)),
16:("Owner Type",lambda:df["Owner_Type"].value_counts().plot(kind="bar",ax=ax)),
17:("Availability",lambda:df["Availability_Status"].value_counts().plot(kind="bar",ax=ax)),
18:("Parking Impact",lambda:sns.boxplot(x="Parking_Space",y="Price_in_Lakhs",data=df,ax=ax)),
19:("Amenities Impact",lambda:sns.boxplot(x="Amenities",y="Price_per_SqFt",data=df,ax=ax)),
20:("Transport Impact",lambda:sns.boxplot(x="Public_Transport_Accessibility",y="Price_per_SqFt",data=df,ax=ax))
}

    title,func=plots[q]
    func()
    plt.xticks(rotation=45)
    return title,fig