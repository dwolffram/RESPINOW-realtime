# Code to generate illustrative figures.

##### Figure contrasting nowcasting, short-term forecasting and scenarios

# Note: this figure is purely illustrative and only loosely based on real data.

# set language to English
Sys.setlocale("LC_TIME", "C")


# get illustration data:
dat <- read.csv(
     here("r", "illustrations", "illustration_data.csv"),
     colClasses = c("date" = "Date")
)

# invent an incomplete part for the coming weeks
incomplete_data <- data.frame(
     date = tail(dat$date, 1) + 7 * (0:2),
     inc7 = tail(dat$inc7, 1) * c(1, 0.95, 0.90)
)

# invent some nowcasts:
nowcast1 <- data.frame(
     date = tail(dat$date, 1) + 7 * (0:4),
     inc7 = tail(dat$inc7, 1) * c(1, 1.08, 1.16, 1.24, 1.36)
)
nowcast1$lower <- nowcast1$inc7 * c(1, 0.97, 0.94, 0.88, 0.8)
nowcast1$upper <- nowcast1$inc7 * c(1, 1.05, 1.1, 1.17, 1.27)

nowcast2 <- nowcast1
nowcast2$inc7 <- nowcast1$inc7 * c(1, 1.02, 1.05, 1.1, 1.15)
nowcast2$lower <- nowcast1$lower * c(1, 1.05, 1.05, 1.1, 1.15)
nowcast2$upper <- nowcast1$upper * c(1, 1.05, 1.05, 1.1, 1.15)

nowcast3 <- nowcast1
nowcast3$inc7 <- nowcast1$inc7 * c(1, 0.95, 0.93, 0.93, 0.93)
nowcast3$lower <- nowcast1$lower * c(1, 0.93, 0.93, 0.93, 0.93)
nowcast3$upper <- nowcast1$upper * c(1, 0.93, 0.93, 0.93, 0.93)


# figure:

pdf(here("figures", "illustration.pdf"), width = 7.5, height = 3.5)

# structure plot area
par(las = 1, mar = c(4, 5, 1, 1))
plot(
     dat$date,
     dat$inc7,
     type = "l",
     xlim = c(as.Date("2020-03-01"), as.Date("2021-01-10")),
     ylim = c(0, 350),
     col = "darkgrey",
     xlab = "",
     ylab = "Daily incidence",
     axes = FALSE
)
axis(
     1,
     at = as.Date(c(
          "2020-03-01",
          "2020-05-01",
          "2020-07-01",
          "2020-09-01",
          "2020-11-01",
          "2021-01-01"
     )),
     labels = c("Mar", "May", "Jul", "Sep", "Nov", "Jan")
)
axis(2)

# rectangle for partial data:
rect(
     as.Date("2020-09-22"),
     -20,
     as.Date("2020-09-22") + c(0, 14),
     450,
     col = "grey95",
     border = NA
)

# delimit area for short-term forecasting
abline(v = as.Date("2020-09-22") + 14, lty = 2, col = "darkgrey")
abline(v = as.Date("2020-09-22") + 28, lty = 2, col = "darkgrey")

# add nowcasts:
# shaded
polygon(
     c(nowcast3$date, rev(nowcast3$date)),
     c(nowcast3$lower, rev(nowcast3$upper)),
     col = rgb(1, 0.8, 0.5, 0.2),
     border = NA
)
polygon(
     c(nowcast2$date, rev(nowcast2$date)),
     c(nowcast2$lower, rev(nowcast2$upper)),
     col = rgb(0, 0.4, 1, 0.2),
     border = NA
)
polygon(
     c(nowcast1$date, rev(nowcast1$date)),
     c(nowcast1$lower, rev(nowcast1$upper)),
     col = rgb(0, 0.8, 0.5, 0.2),
     border = NA
)
# lines
lines(nowcast1$date, nowcast1$inc7, col = rgb(0, 0.8, 0.5))
lines(nowcast2$date, nowcast2$inc7, col = rgb(0, 0.4, 1))
lines(nowcast3$date, nowcast3$inc7, col = rgb(1, 0.8, 0.5))

# points
points(
     nowcast1$date,
     nowcast1$inc7,
     pch = 21,
     col = rgb(0, 0.8, 0.5),
     bg = "white",
     cex = 0.8
)
points(
     nowcast2$date,
     nowcast2$inc7,
     pch = 21,
     col = rgb(0, 0.4, 1),
     bg = "white",
     cex = 0.8
)
points(
     nowcast3$date,
     nowcast3$inc7,
     pch = 21,
     col = rgb(1, 0.8, 0.5),
     bg = "white",
     cex = 0.8
)

# add incomplete data:
points(incomplete_data$date, incomplete_data$inc7, col = "grey80", pch = 20)
points(dat$date, dat$inc7, pch = 20, col = "darkgrey")

# invent and plot scenario curves:
x_scenarios <- tail(nowcast1$date, 1) + 7 * (1:13)
line_red <- c(190, 220, 248, 280, 300, 310, 305, 290, 260, 220, 180, 140, 110)
lines(x_scenarios, line_red, col = "darkred", lwd = 2, lty = 2)

line_orange <- c(
     170,
     180,
     190,
     205,
     220,
     235,
     243,
     250,
     252,
     240,
     230,
     210,
     200
)
lines(x_scenarios, line_orange, col = "darkorange", lwd = 2, lty = 2)

line_purple <- c(150, 160, 175, 185, 190, 187, 180, 165, 150, 135, 115, 95, 75)
lines(x_scenarios, line_purple, col = "purple", lwd = 2, lty = 2)

# add texts:

text(
     as.Date("2020-05-15"),
     320,
     "Available data on previous\n course of the epidemic",
     cex = 0.8
)

lines(c(as.Date("2020-05-10"), as.Date("2020-05-01")), c(280, 150))

text(
     as.Date("2020-07-30"),
     200,
     "Nowcasts serve to\n statistically correct yet\n incomplete recent data.",
     cex = 0.8
)

lines(c(as.Date("2020-09-08"), as.Date("2020-09-26")), c(170, 150))


text(
     as.Date("2020-12-07"),
     50,
     "Longer-term scenario\n projections are conditional\n on specific assumptions.",
     cex = 0.8
)

lines(c(as.Date("2020-11-17"), as.Date("2020-11-25")), c(90, 170))


text(
     as.Date("2020-09-30"),
     330,
     "Short-term forecasts are feasible\n for brief time horizons.",
     cex = 0.8
)

lines(c(as.Date("2020-10-12"), as.Date("2020-10-12")), c(300, 220))

dev.off()


##### Figure illustrating aggregation over nowcast paths

# shorten time series:
dat_short <- subset(dat, date <= as.Date("2020-09-07"))

# invent new incomplete data:
incomplete_data <- data.frame(
     date = tail(dat_short$date, 1) + 7 * (0:4),
     inc7 = tail(dat_short$inc7, 1) * c(1, 0.99, 0.98, 0.95, 0.90)
)

# invent just one nowcast:
nowcast4 <- data.frame(
     date = tail(dat_short$date, 1) + 7 * (0:8),
     inc7 = tail(dat_short$inc7, 1) *
          c(1, 1.01, 1.02, 1.08, 1.16, 1.24, 1.36, 1.5, 1.64)
)
nowcast4$lower <- nowcast4$inc7 *
     c(1, 0.99, 0.98, 0.97, 0.91, 0.89, 0.82, 0.75, 0.68)
nowcast4$upper <- nowcast4$inc7 *
     c(1, 1.01, 1.02, 1.03, 1.07, 1.12, 1.19, 1.29, 1.41)
nowcast4$middle_lower <- (nowcast4$lower + nowcast4$inc7) / 2
nowcast4$middle_upper <- (nowcast4$upper + nowcast4$inc7) / 2

# invent four forecasts (one per displayed nowcast path):
forecast4_lower <- data.frame(
     date = tail(nowcast4$date, 5),
     inc7 = tail(nowcast4$lower, 5)
)
forecast4_lower$lower <- c(1, 0.96, 0.91, 0.86, 0.79) * forecast4_lower$inc7
forecast4_lower$upper <- c(1, 1.04, 1.09, 1.16, 1.25) * forecast4_lower$inc7

forecast4_middle_lower <- data.frame(
     date = tail(nowcast4$date, 5),
     inc7 = tail(nowcast4$middle_lower, 5)
)
forecast4_middle_lower$lower <- c(1, 0.96, 0.91, 0.86, 0.79) *
     forecast4_middle_lower$inc7
forecast4_middle_lower$upper <- c(1, 1.04, 1.09, 1.16, 1.25) *
     forecast4_middle_lower$inc7

forecast4_middle <- data.frame(
     date = tail(nowcast4$date, 5),
     inc7 = tail(nowcast4$inc7, 5)
)
forecast4_middle$lower <- c(1, 0.96, 0.91, 0.86, 0.79) * forecast4_middle$inc7
forecast4_middle$upper <- c(1, 1.04, 1.09, 1.16, 1.25) * forecast4_middle$inc7

forecast4_middle_upper <- data.frame(
     date = tail(nowcast4$date, 5),
     inc7 = tail(nowcast4$middle_upper, 5)
)
forecast4_middle_upper$lower <- c(1, 0.96, 0.91, 0.86, 0.79) *
     forecast4_middle_upper$inc7
forecast4_middle_upper$upper <- c(1, 1.04, 1.09, 1.16, 1.25) *
     forecast4_middle_upper$inc7

forecast4_upper <- data.frame(
     date = tail(nowcast4$date, 5),
     inc7 = tail(nowcast4$upper, 5)
)
forecast4_upper$lower <- c(1, 0.96, 0.91, 0.86, 0.79) * forecast4_upper$inc7
forecast4_upper$upper <- c(1, 1.04, 1.09, 1.16, 1.25) * forecast4_upper$inc7


# Figure:
pdf(here("figures", "illustration2.pdf"), width = 7.5, height = 3.5)

# structure plot area
par(las = 1, mar = c(4, 5, 1, 1), mfrow = 1:2)

# Panel 1
# initialize plot
plot(
     dat_short$date,
     dat_short$inc7,
     type = "l",
     xlim = c(as.Date("2020-08-01"), as.Date("2020-12-05")),
     ylim = c(0, 250),
     col = "darkgrey",
     xlab = "",
     ylab = "Weekly incidence",
     axes = FALSE
)
axis(
     1,
     at = as.Date(c(
          "2020-03-01",
          "2020-05-01",
          "2020-07-01",
          "2020-09-01",
          "2020-11-01",
          "2021-01-01"
     )),
     labels = c("Mar", "May", "Jul", "Sep", "Nov", "Jan")
)
axis(2)

# rectangle for nowcast area
rect(
     as.Date("2020-09-22") - 14,
     -20,
     as.Date("2020-09-22") + 14,
     450,
     col = "grey95",
     border = NA
)

abline(v = as.Date("2020-09-22") + 14, lty = 2, col = "darkgrey")

# nowcast paths:

# lines
lines(head(nowcast4$date, 5), head(nowcast4$inc7, 5), col = rgb(0, 0.4, 1))
lines(head(nowcast4$date, 5), head(nowcast4$lower, 5), col = rgb(0, 0.4, 1))
lines(
     head(nowcast4$date, 5),
     head(nowcast4$middle_lower, 5),
     col = rgb(0, 0.4, 1)
)
lines(
     head(nowcast4$date, 5),
     head(nowcast4$middle_upper, 5),
     col = rgb(0, 0.4, 1)
)
lines(head(nowcast4$date, 5), head(nowcast4$upper, 5), col = rgb(0, 0.4, 1))

# forecast paths:

# shaded:
polygon(
     c(forecast4_lower$date, rev(forecast4_lower$date)),
     c(forecast4_lower$lower, rev(forecast4_lower$upper)),
     col = rgb(0, 0.4, 1, 0.2),
     border = NA
)
polygon(
     c(forecast4_middle_lower$date, rev(forecast4_middle_lower$date)),
     c(forecast4_middle_lower$lower, rev(forecast4_middle_lower$upper)),
     col = rgb(0, 0.4, 1, 0.2),
     border = NA
)
polygon(
     c(forecast4_middle$date, rev(forecast4_middle$date)),
     c(forecast4_middle$lower, rev(forecast4_middle$upper)),
     col = rgb(0, 0.4, 1, 0.2),
     border = NA
)
polygon(
     c(forecast4_middle_upper$date, rev(forecast4_middle_upper$date)),
     c(forecast4_middle_upper$lower, rev(forecast4_middle$upper)),
     col = rgb(0, 0.4, 1, 0.2),
     border = NA
)
polygon(
     c(forecast4_upper$date, rev(forecast4_upper$date)),
     c(forecast4_upper$lower, rev(forecast4_upper$upper)),
     col = rgb(0, 0.4, 1, 0.2),
     border = NA
)

# lines:
lines(
     nowcast4$date,
     nowcast4$inc7,
     pch = 21,
     col = rgb(0, 0.4, 1),
     bg = "white",
     cex = 0.6
)
lines(
     nowcast4$date,
     nowcast4$lower,
     pch = 21,
     col = rgb(0, 0.4, 1),
     bg = "white",
     cex = 0.6
)
lines(
     nowcast4$date,
     nowcast4$upper,
     pch = 21,
     col = rgb(0, 0.4, 1),
     bg = "white",
     cex = 0.6
)
lines(
     nowcast4$date,
     nowcast4$middle_lower,
     pch = 21,
     col = rgb(0, 0.4, 1),
     bg = "white",
     cex = 0.6
)
lines(
     nowcast4$date,
     nowcast4$middle_upper,
     pch = 21,
     col = rgb(0, 0.4, 1),
     bg = "white",
     cex = 0.6
)

points(incomplete_data$date, incomplete_data$inc7, col = "grey80", pch = 20)
lines(incomplete_data$date, incomplete_data$inc7, col = "grey80", pch = 20)
points(dat_short$date, dat_short$inc7, pch = 20, col = "darkgrey")

# text:
text(as.Date("2020-09-21"), 200, "Nowcast\n quantile\n paths", cex = 0.8)

text(
     as.Date("2020-11-19"),
     150,
     "Forecast\n distributions\n per path",
     cex = 0.8
)


# Panel 2
plot(
     dat_short$date,
     dat_short$inc7,
     type = "l",
     xlim = c(as.Date("2020-08-01"), as.Date("2020-12-05")),
     ylim = c(0, 250),
     col = "darkgrey",
     xlab = "",
     ylab = "Weekly incidence",
     axes = FALSE
)
axis(
     1,
     at = as.Date(c(
          "2020-03-01",
          "2020-05-01",
          "2020-07-01",
          "2020-09-01",
          "2020-11-01",
          "2021-01-01"
     )),
     labels = c("Mar", "May", "Jul", "Sep", "Nov", "Jan")
)
axis(2)

# rectangle for nowcast area
rect(
     as.Date("2020-09-22") - 14,
     -20,
     as.Date("2020-09-22") + 14,
     450,
     col = "grey95",
     border = NA
)

abline(v = as.Date("2020-09-22") + 14, lty = 2, col = "darkgrey")

# plot combined nowcast:
# shaded
polygon(
     c(nowcast4$date, rev(nowcast4$date)),
     c(nowcast4$lower, rev(nowcast4$upper)) *
          c(
               1,
               1,
               1,
               1,
               1,
               0.97,
               0.93,
               0.88,
               0.82,
               1.18,
               1.12,
               1.07,
               1.03,
               1,
               1,
               1,
               1,
               1
          ),
     col = rgb(0, 0.4, 1, 0.2),
     border = NA
)
# line:
lines(head(nowcast4$date, 9), head(nowcast4$inc7, 9), col = rgb(0, 0.4, 1))

# points:
points(
     nowcast4$date,
     nowcast4$inc7,
     pch = 21,
     col = rgb(0, 0.4, 1),
     bg = "white",
     cex = 0.6
)

# imcomplete data
points(incomplete_data$date, incomplete_data$inc7, col = "grey80", pch = 20)
lines(incomplete_data$date, incomplete_data$inc7, col = "grey80", pch = 20)
points(dat_short$date, dat_short$inc7, pch = 20, col = "darkgrey")

# text:
text(as.Date("2020-11-21"), 150, "Merged\n forecast\n distribution", cex = 0.8)

dev.off()
