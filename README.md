# GMeow_Indian_rest

This tool is designed to help you make an informed decision about where to open an Indian restaurant in UK cities such as London and Bath by combining real-world public data with financial modelling. It works by scanning the cities in a grid pattern, where each grid point represents a possible location.

For each point, it pulls competitor data from the UK Food Standards Agency (FHRS) database, which lists all registered food businesses along with their type, hygiene rating, and location. It uses demographic and population density data from the UK Office for National Statistics (ONS) to understand the local customer base. For foot traffic estimates, it can incorporate open datasets from Transport for London (TfL) or local councils. Menu prices and cuisine type information are optionally collected from publicly available restaurant menus online (via polite scraping or your own provided datasets). Rent estimates are based on open property listing sites or user-provided assumptions, while operating costs (staff, food, utilities) are calculated using industry-standard percentages.

The tool then runs a Monte Carlo simulation thousands of times for each location, varying factors such as daily customer count, average spend per customer, and cost percentages. This produces a realistic range of possible yearly revenues, net profits, and return on investment (ROI) for that spot. Key metrics in the output include:

median_revenue – the middle value of predicted annual sales

median_roi – the median profitability ratio (profit ÷ initial investment)

payback period – estimated years to recover the initial investment

competitors – number of similar restaurants within a set radius

The outputs are delivered as:

CSV files with all raw data and summary stats per location

PDF reports with easy-to-read summaries for investors

interactive heatmaps showing competition and ROI visually on the city map

graphs and charts (histograms, box plots) to illustrate revenue and profit distributions

In short, this tool turns raw public data into clear, actionable insights about which parts of a city are most promising for opening a profitable Indian restaurant, giving you both the upside potential and the downside risks for each location.
