CREATE DATABASE churn_project;
USE churn_project;

SELECT COUNT(*) AS Total_Rows FROM customer_orders;
SELECT * FROM customer_orders LIMIT 5;

-- Total Business Overview --
SELECT 
    COUNT(DISTINCT Customer_ID) AS Total_Customers,
    COUNT(DISTINCT Order_ID) AS Total_Orders,
    SUM(Revenue) AS Total_Revenue,
    ROUND(AVG(Revenue), 2) AS Avg_Order_Revenue,
    SUM(Quantity) AS Total_Quantity_Sold,
    ROUND(AVG(Quantity), 2) AS Avg_Quantity_Per_Order
FROM customer_orders;

-- Revenue by Category --
SELECT 
    Category,
    COUNT(Order_ID) AS Total_Orders,
    SUM(Revenue) AS Total_Revenue,
    ROUND(AVG(Revenue), 2) AS Avg_Revenue,
    ROUND(SUM(Revenue) * 100.0 / (SELECT SUM(Revenue) FROM customer_orders), 2) AS Revenue_Percentage
FROM customer_orders
GROUP BY Category
ORDER BY Total_Revenue DESC;

-- Revenue by Region --
SELECT 
    Region,
    COUNT(DISTINCT Customer_ID) AS Total_Customers,
    COUNT(Order_ID) AS Total_Orders,
    SUM(Revenue) AS Total_Revenue,
    ROUND(AVG(Revenue), 2) AS Avg_Revenue,
    ROUND(SUM(Revenue) * 100.0 / (SELECT SUM(Revenue) FROM customer_orders), 2) AS Revenue_Percentage
FROM customer_orders
GROUP BY Region
ORDER BY Total_Revenue DESC;

--  Monthly Revenue Trend --
SELECT 
    Order_Month,
    COUNT(DISTINCT Customer_ID) AS Active_Customers,
    COUNT(Order_ID) AS Total_Orders,
    SUM(Revenue) AS Monthly_Revenue,
    ROUND(AVG(Revenue), 2) AS Avg_Order_Value
FROM customer_orders
GROUP BY Order_Month
ORDER BY Order_Month ASC;

-- Top 10 Customers by Revenue --
SELECT 
    Customer_ID,
    COUNT(Order_ID) AS Total_Orders,
    SUM(Revenue) AS Total_Revenue,
    ROUND(AVG(Revenue), 2) AS Avg_Order_Value,
    MIN(Order_Date) AS First_Purchase,
    MAX(Order_Date) AS Last_Purchase
FROM customer_orders
GROUP BY Customer_ID
ORDER BY Total_Revenue DESC
LIMIT 10;

--  Product Performance --
SELECT 
    Product,
    Category,
    COUNT(Order_ID) AS Total_Orders,
    SUM(Quantity) AS Total_Quantity,
    SUM(Revenue) AS Total_Revenue,
    ROUND(AVG(Price), 2) AS Avg_Price,
    ROUND(AVG(Revenue), 2) AS Avg_Revenue
FROM customer_orders
GROUP BY Product, Category
ORDER BY Total_Revenue DESC;

-- Customer Purchase Frequency --
SELECT 
    Frequency_Group,
    COUNT(*) AS Total_Customers,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(DISTINCT Customer_ID) FROM customer_orders), 2) AS Percentage
FROM (
    SELECT 
        Customer_ID,
        COUNT(Order_ID) AS Total_Orders,
        CASE 
            WHEN COUNT(Order_ID) = 1 THEN '1 Order'
            WHEN COUNT(Order_ID) BETWEEN 2 AND 4 THEN '2-4 Orders'
            WHEN COUNT(Order_ID) BETWEEN 5 AND 8 THEN '5-8 Orders'
            ELSE '9+ Orders'
        END AS Frequency_Group
    FROM customer_orders
    GROUP BY Customer_ID
) AS freq
GROUP BY Frequency_Group
ORDER BY Total_Customers DESC;

-- Churn Risk Analysis (RFM with SQL) --
SELECT 
    Customer_ID,
    COUNT(Order_ID) AS Frequency,
    SUM(Revenue) AS Monetary,
    DATEDIFF('2023-12-31', MAX(STR_TO_DATE(Order_Date, '%d-%m-%Y'))) AS Recency_Days,
    CASE 
        WHEN DATEDIFF('2023-12-31', MAX(STR_TO_DATE(Order_Date, '%d-%m-%Y'))) > 90 
        THEN 'High Risk'
        WHEN DATEDIFF('2023-12-31', MAX(STR_TO_DATE(Order_Date, '%d-%m-%Y'))) BETWEEN 60 AND 90 
        THEN 'Medium Risk'
        ELSE 'Low Risk'
    END AS Churn_Risk
FROM customer_orders
GROUP BY Customer_ID
ORDER BY Recency_Days DESC;

-- Churn Risk Summary --
SELECT 
    Churn_Risk,
    COUNT(*) AS Total_Customers,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(DISTINCT Customer_ID) FROM customer_orders), 2) AS Percentage,
    ROUND(AVG(Monetary), 2) AS Avg_Revenue
FROM (
    SELECT 
        Customer_ID,
        SUM(Revenue) AS Monetary,
        CASE 
            WHEN DATEDIFF('2023-12-31', MAX(STR_TO_DATE(Order_Date, '%d-%m-%Y'))) > 90 
            THEN 'High Risk'
            WHEN DATEDIFF('2023-12-31', MAX(STR_TO_DATE(Order_Date, '%d-%m-%Y'))) BETWEEN 60 AND 90 
            THEN 'Medium Risk'
            ELSE 'Low Risk'
        END AS Churn_Risk
    FROM customer_orders
    GROUP BY Customer_ID
) AS risk
GROUP BY Churn_Risk
ORDER BY Total_Customers DESC;