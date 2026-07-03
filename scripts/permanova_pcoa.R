library(vegan)
library(ape)
library(ggplot2)
library(tidyverse)
library(ggpubr)

#setwd("/vol/projects/khuang/downstream_analysis/rheumavor_revision/covariables_adjustment")
#Replace the above line with your working directory path if needed.

##### configuration ######
psa_enthesis <- read.csv("PsA_enthesitis.tsv", sep = "\t", header = T)
mat_psa <- read.csv("mpa4_merged_relab_PsA_sgb_md_final_samples_4vegan_matrix.tsv", sep = "\t", header = T)
md_psa <- read.csv("mpa4_merged_relab_PsA_sgb_md_final_samples_4vegan_metadata.tsv", sep = "\t", header = T)

## Append columns from `psa_enthesis` onto `md_psa` using a shared key column
key_col <- "sample" 

stopifnot(key_col %in% names(md_psa), key_col %in% names(psa_enthesis))

cols_to_add <- setdiff(names(psa_enthesis), key_col)

md_psa <- md_psa %>%
  left_join(psa_enthesis %>% dplyr::select(dplyr::all_of(c(key_col, cols_to_add))),
            by = key_col)


mat_ra <- read.csv("mpa4_merged_relab_RA_sgb_md_final_samples_4vegan_matrix.tsv", sep = "\t", header = T)
md_ra <- read.csv("mpa4_merged_relab_RA_sgb_md_final_samples_4vegan_metadata.tsv", sep = "\t", header = T)

mat_psa_humann <- read.csv("humann3_aggre_pathways_relab_initial_PsA_final_samples_4vegan_matrix.tsv", sep = "\t", header = T)
md_psa_humann <- read.csv("humann3_aggre_pathways_relab_initial_PsA_final_samples_4vegan_metadata.tsv", sep = "\t", header = T)

mat_ra_humann <- read.csv("humann3_aggre_pathways_relab_initial_RA_final_samples_4vegan_matrix.tsv", sep = "\t", header = T)
md_ra_humann <- read.csv("humann3_aggre_pathways_relab_initial_RA_final_samples_4vegan_metadata.tsv", sep = "\t", header = T)


## Append columns from `psa_enthesis` onto `md_psa_humann` using a shared key column
key_col <- "sample" 

stopifnot(key_col %in% names(md_psa_humann), key_col %in% names(psa_enthesis))

cols_to_add <- setdiff(names(psa_enthesis), key_col)

md_psa_humann <- md_psa_humann %>%
  left_join(psa_enthesis %>% dplyr::select(dplyr::all_of(c(key_col, cols_to_add))),
            by = key_col)

##### configuration ######

pcoa_plot <- function(coordis_df, category, fsize = 11, dsize = 1, fstyle = "Arial", to_rm = NULL) {
  # coordis_df: the dataframe containing principal coordinate estimates
  # fsize: the font size
  # dsize: the dot size
  # fstyle: the font style
  # category: specify the category name for separating groups
  # this function is to draw pcoa plot with confidence ellipse
  
  if (is.null(to_rm)) {
    coordis_df <- coordis_df[!(is.na(coordis_df[, category]) | coordis_df[, category] == ""), ]
  }
  else {
    coordis_df <- coordis_df[!(is.na(coordis_df[, category]) | coordis_df[, category] == "" | coordis_df[, category] %in% to_rm), ]
  }
  eval(substitute(ggplot(coordis_df,aes(Axis.1, Axis.2, color = c)),list(c = as.name(category)))) +
    geom_point(size = dsize) + 
    theme_bw() +
    eval(substitute(geom_polygon(stat = "ellipse", aes(fill = c), alpha = 0.1, type = "norm"), list(c = as.name(category)))) +
    labs(x = "PC1", y = "PC2") +
    theme(text = element_text(size = fsize, family = fstyle)) +
    theme(legend.position="bottom") + 
    scale_color_manual(values = c("inefficiency" = "#FFC20A",
                                  "remission" = "#0C7BDC"))
  
}

generate_coordis_df <- function(mat, md) {
  # mat: the loaded matrix from mpa-style dataframe.
  # md: the dataframe containing metadata.
  # this function is to prepare metadata-added coordinates dataframe.
  bray_dist <- vegdist(mat, "bray")
  pcos <- as.data.frame(pcoa(bray_dist)$vectors)
  p_df <- cbind(pcos, md)
  p_df
}

est_permanova <- function(mat, md, tcol, e_vector = NULL, nper = 999, to_rm = NULL){
  if (is.null(to_rm)) {
    clean_md <- md[!(is.na(md[, tcol]) | md[, tcol] == ""), ]
  } else {
    clean_md <- md[!(is.na(md[, tcol]) | md[, tcol] == "" | md[, tcol] %in% to_rm), ]
  }
  clean_idx = rownames(clean_md)
  clean_mat <- mat[rownames(mat) %in% clean_idx, ]
  if (is.null(e_vector)) {
    est <- eval(substitute(adonis2(mat ~ cat, data = md, permutations = nper), list(cat = as.name(tcol))))
  } else {
    mat_char <- deparse(substitute(mat))
    str1 <- paste0(c(paste0(e_vector, collapse = " + ")), tcol, collapse = " + ")
    str2 <- paste0(c(mat_char, str1), collapse = " ~ ")
    print(str2)
    est <- adonis2(eval(parse(text = str2)), data = md, permutations = nper)
  }
  est
}

###### execution section #######
#t_mat_ra_humann <- sin(mat_ra_humann)**2


coordis_psa_df <- generate_coordis_df(mat_psa_humann, md_psa_humann)
coordis_ra_df <- generate_coordis_df(mat_ra_humann, md_ra_humann)

psa_pcoa <- pcoa_plot(coordis_psa_df, "MTX_response", fstyle = "Arial")
ra_pcoa <- pcoa_plot(coordis_ra_df, "MTX_response", fstyle = "Arial")



ggarrange(psa_pcoa, ra_pcoa,
          nrow = 1, ncol = 2)

b_dist <- function(mat) {
  # mat: the loaded matrix from mpa-style dataframe.
  # md: the dataframe containing metadata.
  # this function is to prepare metadata-added coordinates dataframe.
  bray_dist <- vegdist(mat, "bray")
  bray_dist
}

View(md_psa)

est_permanova(mat_psa, md_psa, "MTX_response", c("age", "enthesitis"),
              nper = 999, to_rm = NULL)

est_permanova(mat_psa_humann, md_psa_humann, "MTX_response", c("age", "enthesitis"),
              nper = 999, to_rm = NULL)

adonis2(mat_test ~ MTX_response, data = md_ra, permutations = 999)


