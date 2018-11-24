library(tidyverse)
library(fs)
library(cowplot)

# library(extrafont)
# loadfonts()
# windowsFonts(LMRoman = windowsFont('LM Roman 10'))

# XZ_labeller <- function(value) parse(text = str_c('X = Z == ', value))
# EO_labeller <- function(value) parse(text = str_c('E = O == ', value))
log_breaker <- scales::trans_breaks('log10', function(x) 10^x, n = 3)
log_labeller <- scales::trans_format('log10', function(x) parse(text = ifelse(x == 0, '1', ifelse(x == 1, '10', paste0('10^', x)))))
methods_breaks <- c('basic-min',
                    'extended-min (P=5)',
                    'extended-min-ub (w=Inf)',
                    'extended-min-ub (w=2)')
methods_labels <- c(parse(text = '"basic-min"'),
                    parse(text = str_c('"extended-min "', '(P==5)')),
                    parse(text = str_c('"extended-min-ub "', '(w==infinity)')),
                    parse(text = str_c('"extended-min-ub "', '(w==2)')))

plot_results <- function(df) {
    df2 <- df %>%
        group_by(E, X, Cres, method) %>%
        summarize(time_mean = mean(time),
                  time_median = median(time),
                  time_sd = mad(time)) %>%
        ungroup()

    ggplot(df, aes(x = factor(Cres), y = time, fill = method)) +
        facet_grid(rows = vars(E), cols = vars(X),
                   # labeller = labeller(E = EO_labeller, X = XZ_labeller),
                   labeller = label_bquote(rows = {"|E|" == "|O|"} == .(E),
                                           cols = {"|X|" == "|Y|"} == .(X)),
                   scales = 'free_x') +
        geom_line(data = df2,
                  aes(x = as.numeric(factor(Cres)), y = time_median,
                      color = method, linetype = method),
                  position = position_dodge(width = 0.5),
                  size = 0.5, alpha = 0.8, show.legend = FALSE) +
        geom_jitter(aes(color = method),
                    position = position_jitterdodge(jitter.width = 0.1,
                                                    dodge.width = 0.5),  # 0.8
                    size = 0.5, alpha = 0.3, show.legend = FALSE) +
        geom_boxplot(alpha = 0.6, lwd = 0.2, outlier.shape = NA,
                     position = position_dodge(width = 0.5)) +  # no width
        geom_text(data = tibble(x = c(3, 4, 1, 3),
                                y = c(1, 35, 120, 160),
                                X = c(4, 4, 4, 4),
                                E = c(5, 5, 5, 5),
                                angle = c(0, 0, 32, 18),
                                label = c('basic-min',
                                          'ext-min (P=5)',
                                          'ext-min-ub (w=Inf)',
                                          'ext-min-ub (w=2)'),
                                color = methods_breaks),
                  aes(x = x, y = y, angle = angle,
                      label = label, color = color, fill = NULL),
                  size = 2.6, show.legend = FALSE, hjust = 0, vjust = 0) +
        # geom_violin(alpha = 0.6,
        #             position = position_dodge(width = 0.5)) +
        # hue_pal()(3) = '#F8766D'(red) '#00BA38'(green) '#619CFF'(blue)
        # scale_discrete_manual(aesthetics = c('colour', 'fill'),
        #                       values = c('basic-min' = '#F8766D',
        #                                  'extended-min-ub (w=2)' = '#619CFF')) +
        guides(color = 'none') +
        scale_x_discrete() +
        scale_y_log10(limits = c(NA, 2000),
                      breaks = log_breaker,
                      labels = log_labeller) +
        scale_fill_discrete(name = 'Method',
                            breaks = methods_breaks,
                            labels = methods_labels) +
        scale_linetype_manual(values = c('dotdash', 'solid', 'dotted', 'dashed')) +
        # scale_linetype_manual(values = c('dashed', 'dotted', 'solid')) +
        # coord_cartesian(ylim = c(min(df$time), 2000)) +
        theme_bw() +
        theme(legend.position = 'bottom',
              legend.margin = margin(0, 0, 0, 0),
              legend.box.margin = margin(-5, 0, 0, 0)) +
        labs(x = 'Number of states C',
             y = 'Time, s',
             # title = 'Solving time distributions',
             fill = 'Method')
}


# B10 C36 results
{
sims <- 'sims_C3-6_E1-2_X2-4_P5'
resC36_basic_min <- read_csv(str_interp('_results_merged_${sims}_extended-min-ub-w2.csv')) %>%
    mutate(method = 'basic-min')
resC36_extended_min <- read_csv(str_interp('_results_merged_${sims}_extended-min.csv')) %>%
    mutate(method = 'extended-min (P=5)')
resC36_extended_min_ub <- read_csv(str_interp('_results_merged_${sims}_extended-min-ub.csv')) %>%
    mutate(w = as.integer(w)) %>%
    mutate(method = 'extended-min-ub (w=Inf)')
resC36_extended_min_ub_w2 <- read_csv(str_interp('_results_merged_${sims}_extended-min-ub-w2.csv')) %>%
    mutate(method = 'extended-min-ub (w=2)')
resC36 <- bind_rows(
    resC36_basic_min,
    resC36_extended_min,
    resC36_extended_min_ub,
    resC36_extended_min_ub_w2
)
rm(sims)
}


# B50 C37 results
{
sims <- 'sims_B50_C3-7_E1-2-5_X2-4-8_P5'
resC37_basic_min <- read_csv(str_interp('_results_merged_${sims}_basic-min.csv')) %>%
    mutate(method = 'basic-min')
resC37_extended_min <- read_csv(str_interp('_results_merged_${sims}_extended-min.csv')) %>%
    mutate(method = 'extended-min (P=5)')
resC37_extended_min_ub <- read_csv(str_interp('_results_merged_${sims}_extended-min-ub.csv')) %>%
    mutate(w = as.integer(w)) %>%  # all null -> NA_character_
    mutate(method = 'extended-min-ub (w=Inf)')
resC37_extended_min_ub_w2 <- read_csv(str_interp('_results_merged_${sims}_extended-min-ub-w2.csv')) %>%
    mutate(method = 'extended-min-ub (w=2)')
resC37 <- bind_rows(
    resC37_basic_min,
    resC37_extended_min,
    resC37_extended_min_ub,
    resC37_extended_min_ub_w2
) %>% filter(Cres >= 3)
rm(sims)
}


# B6 C37 results
{
sims <- 'sims_B6_C3-7_E1-2-5_X2-4-8_P5'
resB6C37_basic_min <- read_csv(str_interp('_results_merged_${sims}_basic-min.csv')) %>%
    mutate(method = 'basic-min')
resB6C37_extended_min <- read_csv(str_interp('_results_merged_${sims}_extended-min.csv')) %>%
    mutate(method = 'extended-min (P=5)')
resB6C37_extended_min_ub <- read_csv(str_interp('_results_merged_${sims}_extended-min-ub.csv')) %>%
    mutate(w = as.integer(w)) %>%  # all null -> NA_character_
    mutate(method = 'extended-min-ub (w=Inf)')
resB6C37_extended_min_ub_w2 <- read_csv(str_interp('_results_merged_${sims}_extended-min-ub-w2.csv')) %>%
    mutate(Pres = as.integer(Pres)) %>%
    mutate(Nres = as.integer(Nres)) %>%
    mutate(method = 'extended-min-ub (w=2)')
resB6C37 <- bind_rows(
    resB6C37_basic_min,
    resB6C37_extended_min,
    resB6C37_extended_min_ub,
    resB6C37_extended_min_ub_w2
) %>% filter(Cres >= 3)
rm(sims)
}


res_merged <- bind_rows(
    resC36,
    resC37,
    resB6C37
) %>%
    # filter(method != 'extended-min (P=5)') %>%
    filter(Cres >= 3)


plot_results(res_merged)
ggsave('plot_solve-time-distribution_merged.pdf', width = 7, height = 6)


# Total guards size distribution
# res_merged %>%
#     filter(!is.na(Nres)) %>%
#     ggplot(aes(x = factor(Cres), y = Nres, fill = method)) +
#     facet_grid(rows = vars(E), cols = vars(X),
#                # labeller = labeller(E = EO_labeller, X = XZ_labeller),
#                labeller = label_bquote(rows = {E == O} == .(E),
#                                        cols = {X == Y} == .(X)),
#                scales = 'free_x') +
#     geom_jitter(aes(color = method),
#                 position = position_jitterdodge(jitter.width = 0.1,
#                                                 dodge.width = 0.45),  # 0.8
#                 alpha = 0.5, show.legend = FALSE) +
#     geom_boxplot(outlier.shape = NA, alpha = 0.7,
#                  lwd = 0.3,
#                  position = position_dodge(width = 0.5)) +  # no width
#     guides(color = 'none') +
#     scale_x_discrete() +
#     scale_y_log10(limits = c(NA, 200),
#                   breaks = log_breaker,
#                   labels = log_labeler) +
#     theme_bw() +
#     theme(title = element_text(size = 12),
#           legend.position = 'bottom') +
#     labs(title = str_interp('Guard size distributions'),
#          fill = 'Method',
#          x = 'Number of states C',
#          y = 'Number of nodes N')


# Average guard size distribution
# res_merged %>%
#     filter(!is.na(Nres)) %>%
#     select(Cres, method, Nres, Tres) %>%
#     mutate(NperT = Nres / Tres) %>%
#     ggplot(aes(x = factor(Cres), y = NperT, fill = method)) +
#     geom_jitter(aes(color = method),
#                 position = position_jitterdodge(jitter.width = 0.1,
#                                                 dodge.width = 0.75),
#                 alpha = 0.5, show.legend = FALSE) +
#     geom_boxplot(alpha = 0.5, lwd = 0.3, outlier.shape = NA,
#                  position = position_dodge2()) +
#     scale_x_discrete() +
#     scale_y_continuous() +
#     scale_fill_discrete(name = 'Method',
#                         breaks = methods_breaks,
#                         labels = methods_labels) +
#     coord_cartesian(ylim = c(1, 3.5)) +
#     theme_bw() +
#     # theme(text = element_text(family = 'LMRoman')) +
#     labs(title = 'Distributions of average guard size',
#          x = 'Number of states C',
#          y = 'Guard size / number of transitions')
# ggsave('plot_average-guard-size-distribution_merged_C36_C37.pdf', width = 6, height = 3)


resC37 %>%
    mutate(path = str_c(path, '/C37')) %>%
    bind_rows(resC36) %>%
    filter(!is.na(Nres)) %>%
    filter(str_detect(method, 'w')) %>%
    # select(path, method, Nres) %>%
    select(path, Cres, method, Nres, Tres) %>%
    # mutate(NperT = Nres / Tres) %>%
    spread(method, Nres) %>%
    filter(!is.na(`extended-min-ub (w=Inf)`) & !is.na(`extended-min-ub (w=2)`)) %>%
    filter(`extended-min-ub (w=Inf)` != `extended-min-ub (w=2)`)


res_merged %>%
    select(path, method, time) %>%
    arrange(desc(time))

res_merged %>%
    count(method)
res_merged %>%
    count()
