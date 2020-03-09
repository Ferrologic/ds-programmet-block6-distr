library(tidymodels)
#library(drake)
library(tidyverse)
library(magrittr)
## step 1 -- read data.
read_csv("block6/hemnet/hemnet_data.csv") -> hemnet


set.seed(1)

hemnet <- hemnet %>%
  select(
    final_price,
    type,
    city,
    sq_m,
    rooms,
    fee,
    add_area_sqm,
    list_price
  ) %>%
  mutate_at("add_area_sqm", list(~if_else(is.na(.), 0, .))) %>%
  drop_na()


hemnet %>%
  ggplot(aes(x = type, y = final_price)) +
  #geom_jitter(aes(color = bid_premium, size = rooms, alpha = bid_premium)) +
  geom_violin() +
  scale_size(range = c(2,6)) +
  #scale_alpha(range = c(0.2,0.9)) +
  coord_flip() +
  theme_minimal()


sale_split <- initial_split(hemnet, prop = 0.80)

sale_split %>%
  training() %>%
  glimpse()


#%>%
#  step_other(Neighborhood, threshold = 0.01,
#             other = "other") %>%
#  step_other(Screen_Porch, threshold = 0.1,
#             other = ">0")

#?step_discretize

sale_recipe <- NULL
training(sale_split) %>%
  recipe(final_price ~ .) %>%
  #step_YeoJohnson(all_numeric()) %>%
  step_discretize(add_area_sqm, fee,
                  options = list(na.rm = T, keep_na = T)) %>%
  #step_sqrt(list_price_sqm) %>%
  step_poly(list_price, degree = 2) %>%
  step_log(sq_m) %>%
  step_string2factor(all_nominal(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes()) %>%
  step_corr(all_predictors(), threshold = 0.5) %>%
  step_center(all_predictors(), -all_outcomes()) %>%
  step_scale(all_predictors(), -all_outcomes()) %>%
  step_nzv(all_predictors()) %>%
  prep() -> sale_recipe

sale_recipe %>%
  bake(testing(sale_split)) -> sale_testing

sale_training <- juice(sale_recipe)

sale_training %>% glimpse()

#?boost_tree

xg_fit <- boost_tree(learn_rate=0.8,
                      trees = 300,
                      tree_depth= 14,
                      min_n=1,
                      sample_size=1,
                      mode="regression") %>%
  set_engine("xgboost", verbose=1) %>%
  fit(final_price ~ ., data = sale_training)


predict(xg_fit, new_data = sale_testing) -> pred_xg_train
## get prediction on test set
predict(xg_fit, new_data = sale_testing) -> pred_xg_test
## get probabilities on test set
predict(xg_fit, new_data = sale_testing) -> prob_xg_test

bind_cols(sale_testing, pred_xg_test) %>%
  metrics(., truth = final_price, estimate = .pred)


metric_set(accuracy,
           bal_accuracy,
           sens,
           yardstick::spec,
           precision,
           recall,
           ppv,
           npv) -> multimetric

bind_cols(sale_testing, pred_xg_test) %>%
  multimetric(., truth = final_price, estimate = .pred)

bind_cols(sale_testing, pred_xg_test)

bind_cols(sale_testing, prob_xg_test) %>%
  roc_auc(., truth = bid_premium, .pred_Yes)

bind_cols(sale_testing, prob_xg_test) %>%
 roc_curve(., truth = bid_premium, .pred_Yes) -> roc_data

roc_data %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_path() +
  geom_abline(lty = 3) +
  coord_equal() +
  theme_minimal()


#########

### tune + cv should be done. avg param accuracy.

cv_train <- vfold_cv(sale_training,
                     v = 5,
                     repeats = 5,
                     strata = "bid_premium")

xgmod <-
  boost_tree(learn_rate=0.1,
             trees = 300,
             tree_depth= 14,
             min_n=1,
             sample_size=1,
             mode="classification") %>%
  set_engine("xgboost", verbose=1)


## compute mod on kept part
cv_fit <- function(splits, mod, ...) {
  res_mod <-
    fit(mod, bid_premium ~ ., data = analysis(splits)) #, family = binomial)
  return(res_mod)
}

## get predictions on holdout sets
cv_pred <- function(splits, mod){
  # Save the 10%
  holdout <- assessment(splits)
  pred_assess <- bind_cols(truth = holdout$bid_premium, predict(mod, new_data = holdout))
  return(pred_assess)
}

## get probs on holdout sets
cv_prob <- function(splits, mod){
  holdout <- assessment(splits)
  prob_assess <- bind_cols(truth = as.factor(holdout$bid_premium),
                           predict(mod, new_data = holdout, type = "prob"))
  return(prob_assess)
}

res_cv_train <-
  cv_train %>%
  mutate(res_mod = map(splits, .f = cv_fit, xgmod), ## fit model
         res_pred = map2(splits, res_mod, .f = cv_pred), ## predictions
         res_prob = map2(splits, res_mod, .f = cv_prob)) ## probabilities


res_cv_train %>%
  mutate(metrics = map(res_pred,
                       multimetric,
                       truth = truth,
                       estimate = .pred_class)) %>%
  unnest(metrics) %>%
  ggplot() +
  aes(x = id, y = .estimate) +
  geom_point(aes(color = .estimate)) +
  facet_wrap(~ .metric, scales = "free_y") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30))

res_cv_train %>%
  mutate(roc = map(res_prob, roc_curve, truth = truth, .pred_Yes)) %>%
  unnest(roc) %>%
  ggplot() +
  aes(x = 1 - specificity, y = sensitivity, color = id2) +
  geom_path() +
  geom_abline(lty = 3) +
  facet_wrap(~id)


library(probably)


res_cv_train %>%
  unnest(res_prob) %>%
  select(-splits, -res_mod, -res_pred) %>%
  mutate(.pred = make_two_class_pred(.pred_No, levels(truth), threshold = .5)) -> hard_pred_0.5


hard_pred_0.5 %>%
  count(.truth = truth, .pred)

threshold_data <- hard_pred_0.5 %>%
  threshold_perf(truth, .pred_No, thresholds = seq(0.1, 1, by = 0.025))


threshold_data <- threshold_data %>%
  filter(.metric != "distance") %>%
  mutate(group = case_when(
    .metric == "sens" | .metric == "spec" ~ "1",
    TRUE ~ "2"
  ))

max_j_index_threshold <- threshold_data %>%
  filter(.metric == "j_index") %>%
  filter(.estimate == max(.estimate)) %>%
  pull(.threshold)

ggplot(threshold_data, aes(x = .threshold, y = .estimate, color = .metric, alpha = group)) +
  geom_line() +
  theme_minimal() +
  scale_color_viridis_d(end = 0.9) +
  scale_alpha_manual(values = c(.4, 1), guide = "none") +
  geom_vline(xintercept = max_j_index_threshold, alpha = .6, color = "grey30") +
  labs(
    x = "'Yes' Threshold\n(above this value is considered 'bid premium = yes')",
    y = "Metric Estimate",
    title = "Balancing performance by varying the threshold",
    subtitle = "Sensitivity or specificity alone might not be enough!\nVertical line = Max J-Index"
  )

threshold_data %>%
  filter(.threshold == max_j_index_threshold) %>%
  slice(1) %>%
  pull(.threshold) -> threshhold

res_cv_train %>%
  unnest(res_prob) %>%
  select(-splits, -res_mod, -res_pred) %>%
  mutate(.pred = make_two_class_pred(.pred_No, levels(truth), threshold = threshhold)) %>%
  mutate(prediction = .pred %>% as.factor()) -> opt_thresh_preds


opt_thresh_preds %>%
  multimetric(., truth = truth, estimate = prediction)

#vignette("equivocal-zones", "probably")
## find max uncertain zones. when the model is certain -- use preds. otherwise -> uncertain. evaluate optimal thresholds.

### reportable rate

opt_thresh_preds -> pred_tbl

pred_tbl

pred_tbl %>%
  mutate(
    .pred = make_two_class_pred(
      estimate = .pred_No,
      levels = levels(truth),
      threshold = 0.35,
      buffer = 0.05
    )
  ) -> segment_pred

segment_pred

segment_pred %>%
  count(.pred)

segment_pred %>%
  summarise(reportable = reportable_rate(.pred))

segment_pred %>%
  mutate(.pred_fct = as.factor(.pred)) %>%
  count(.pred, .pred_fct)


segment_pred %>%
  mutate(.pred_fct = .pred %>% as.factor) %>%
  multimetric(., truth = truth, estimate = .pred_fct)

segment_pred %>%
  mutate(.pred_fct = .pred %>% as.factor) -> reporting_stats

###### different levels of buffer & results

buffer_zone = seq(0,0.4,0.01)
i = 1
lapply(1:length(buffer_zone), function(i){

  segment_pred <- pred_tbl %>%
    mutate(
      .pred = make_two_class_pred(
        estimate = .pred_No,
        levels = levels(truth),
        threshold = 0.35,
        buffer = buffer_zone[i]
      )
    )

  segment_pred %>%
    summarise(reportable = reportable_rate(.pred)) -> rep_rate

  segment_pred %>%
    mutate(.pred_fct = as.factor(.pred)) -> reporting_stats

  reporting_stats %>%
    multimetric(., truth = truth, estimate = .pred_fct) %>%
    select(-.estimator) %>%
    bind_rows(
      tibble(
        .metric = "reportable_rate",
        .estimate = rep_rate %>% pull(reportable)
      )
    ) %>%
    mutate(buffer = rep(buffer_zone[i]))

}) %>%
  bind_rows() -> buffer_tbl

buffer_tbl

buffer_tbl %>%
  ggplot(aes(x = buffer, y = .estimate, color = .metric, linetype = .metric)) +
  geom_point(size = 1) +
  geom_line(alpha = 1) +
  xlab("Buffer size (+/- from optimal p-threshold)") +
  ylab("Estimate") +
  ggtitle("Accuracy metrics & reportable rates",
          "Given various equivocal zones") +
  theme_minimal() -> bt

bt %>% plotly::ggplotly()

buffer_tbl

chosen_buffer = 0.15

opt_pred <- pred_tbl %>%
  mutate(
    .pred = make_two_class_pred(
      estimate = .pred_No,
      levels = levels(truth),
      threshold = 0.35,
      buffer = chosen_buffer
    )
  )

opt_pred %<>%
  mutate(.pred_fct = as.factor(.pred)) %>%
  na.omit()

opt_pred %>%
  multimetric(., truth = truth, estimate = .pred_fct) -> acc_mets


paste0(acc_mets %>%
         colnames %>%
         paste(acc_mets %>%
                 as.matrix() %>%
                 as.vector, collapse = "\n"), "\n",
       paste( "Using p-threshold:",max_j_index_threshold, "and buffer zone of", chosen_buffer), collapse = "")  -> text

text

opt_pred %>%
  conf_mat(truth = truth, estimate = .pred_fct) %>%
  pluck(1) %>%
  as_tibble() %>%
  ggplot(aes( Truth, Prediction, fill = n)) +
  geom_tile( show.legend = FALSE) +
  geom_text(aes(label = n), color = "white", alpha = 1, size = 8) +
  scale_fill_viridis_c() +
  theme_minimal() +
  labs(
    title = "Confusion matrix",
    subtitle = text
  ) +
  theme(panel.grid.major.x = ggplot2::element_blank(),
        panel.grid.major.y = ggplot2::element_blank())


############################## predictioNs

??conf_mat


#########################################################################
## tune simple

#xgb_model

xgb_mod <- boost_tree(learn_rate=tune(),
                      trees = tune(),
                      tree_depth= tune(),
                      min_n= tune(),
                      sample_size= tune(),
                      mode="classification") %>%
  set_engine("xgboost", verbose=0)

xgb_mod <- boost_tree(learn_rate=0.8,
           trees = tune(),
           tree_depth= 14,
           min_n=1,
           sample_size=1,
           mode="classification") %>%
  set_engine("xgboost", verbose=1)

bid_rs <- bootstraps(sale_training, times = 1)

bid_rs

roc_vals <- metric_set(roc_auc)
roc_vals

ctrl <- control_grid(verbose = T)

ctrl

grid_form <-
  tune_grid(
    bid_premium ~ .,
    model = xgb_mod,
    resamples = bid_rs,
    metrics = roc_vals,
    control = ctrl
  )

grid_form

grid_form %>% select(.metrics) %>% slice(1) %>% pull(1)
estimates <- collect_metrics(grid_form)
estimates

#5.63e-2

#tune

show_best(grid_form)



## 10-fold cross validation
folds <- vfold_cv(training(sale_split), v = 5)

folds %>%
  mutate(
    recipes = splits %>%
      # Prepper is a wrapper for `prep()` which handles `split` objects
      map(prepper, recipe = sale_recipe),
    test_data = splits %>% map(analysis),
    rf_fits =
      map2(
        recipes,
        test_data,
        ~ fit(
          xgb_model,
          formula(.x),
          data = bake(object = .x, new_data = .y)
        )
      )
  ) -> folded


folded

predict_xgb <- function(split, rec, model) {
  test <- bake(sale_recipe %>% prep(), assessment(split))
  tibble(
    actual = test$bid_premium,
    predicted = predict(model, test, type = "prob")
  )
}

predictions <-
  folded %>%
  mutate(
    pred =
      list(
        splits,
        recipes,
        rf_fits
      ) %>%
      pmap(predict_xgb)
  )

predictions %>%
  as_tibble() %>%
  select(id, pred) %>%
  unnest(pred) %>%
  mutate(.pred = make_two_class_pred(predicted$.pred_No, levels(actual), threshold = .5)) -> ls_tbl

ls_tbl %>% select(predicted) %>% pluck(1) %>%
  bind_cols(ls %>% select(id, actual, .pred)) %>%
  select(id, everything()) %>%
  mutate(predicted = .pred %>% as.factor) %>%
  nest(data = c(.pred_No, .pred_Yes, actual, .pred, predicted)) %>%
  mutate(
    metrics = data %>% map(~ metrics(., truth = actual, estimate = predicted))
  ) %>%
  select(metrics) %>%
  unnest(metrics)

#### tidy eval -- opt thresh / equiovical zones by fold.

ls_tbl


############




#sale_ranger <- rand_forest(trees = 100, mode = "classification") %>%
#  set_engine("ranger") %>%
#  fit(bid_premium ~ ., data = sale_training)
#install.packages("randomForest")


bt_model <- NULL



sale_rf <-  rand_forest(trees = 100, mode = "classification") %>%
  set_engine("randomForest") %>%
  fit(bid_premium ~ ., data = sale_training)



#sale_ranger %>%
#  predict(sale_testing) %>%
#  bind_cols(sale_testing) %>%
#  glimpse()


#sale_ranger %>%
#  predict(sale_testing) %>%
#  bind_cols(sale_testing) %>%
#  metrics(truth = Species, estimate = .pred_class)

#sale_rf %>%
#  predict(sale_testing) %>%
#  bind_cols(sale_testing) %>%
#  metrics(truth = bid_premium, estimate = .pred_class) #-> prediction

bt_model$fit

tidy_importance <- function(xgb_model, num_features = 0.2){


  xgb.importance(model=bt_model$fit) %>%
    as_tibble() %>%
    top_n(0.2*n(), Gain) -> plot_obj

  plot_obj %>%
    ggplot(aes(x = reorder(Feature, Gain), y = Gain, fill = Gain)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    labs(
      x = "Gain",
      y = "Feature",
      title = "Feature Importance"
    ) +
    theme_custom +
    theme(legend.position = "none")

}

bt_model %>%
  tidy_importance()






predict(bt_model, sale_testing, type = "prob") %>%
  bind_cols(predict(bt_model, sale_testing)) %>%
  bind_cols(select(sale_testing, bid_premium)) -> prediction_tbl

#prediction_tbl %>%
#  filter(.pred_class != bid_premium)


bt_model %>%
  predict(sale_testing) %>%
  bind_cols(sale_testing) %>%
  metrics(truth = bid_premium, estimate = .pred_class) %>%
  bind_rows(
    bt_model %>%
      predict(sale_testing) %>%
      bind_cols(sale_testing) %>%
      sens(truth = bid_premium, estimate = .pred_class) %>%
      bind_rows(
        bt_model %>%
          predict(sale_testing) %>%
          bind_cols(sale_testing) %>%
          yardstick::spec(truth = bid_premium, estimate = .pred_class)
      )
  ) -> mets

mets

#install.packages("probably")
library(probably)

#vignette("where-to-use", "probably")

predictions <- bt_model %>%
  predict(new_data = sale_testing, type = "prob")

predictions

premium_test_pred <- bind_cols(predictions, sale_testing)

#premium_test_pred

hard_pred_0.5 <- premium_test_pred %>%
  mutate(.pred = make_two_class_pred(.pred_No, levels(bid_premium), threshold = .5)) %>%
  select(bid_premium, contains(".pred"))

hard_pred_0.5 %>%
  count(.truth = bid_premium, .pred)

threshold_data <- premium_test_pred %>%
  threshold_perf(bid_premium, .pred_No, thresholds = seq(0.1, 1, by = 0.025))


threshold_data <- threshold_data %>%
  filter(.metric != "distance") %>%
  mutate(group = case_when(
    .metric == "sens" | .metric == "spec" ~ "1",
    TRUE ~ "2"
  ))

max_j_index_threshold <- threshold_data %>%
  filter(.metric == "j_index") %>%
  filter(.estimate == max(.estimate)) %>%
  pull(.threshold)

ggplot(threshold_data, aes(x = .threshold, y = .estimate, color = .metric, alpha = group)) +
  geom_line() +
  theme_minimal() +
  scale_color_viridis_d(end = 0.9) +
  scale_alpha_manual(values = c(.4, 1), guide = "none") +
  geom_vline(xintercept = max_j_index_threshold, alpha = .6, color = "grey30") +
  labs(
    x = "'Yes' Threshold\n(above this value is considered 'bid premium = yes')",
    y = "Metric Estimate",
    title = "Balancing performance by varying the threshold",
    subtitle = "Sensitivity or specificity alone might not be enough!\nVertical line = Max J-Index"
  )

threshold_data %>%
  filter(.threshold == max_j_index_threshold)

##### test this

premium_test_pred %>%
  select(1:2,5) %>%
  mutate(prediction = ifelse(.pred_No >= max_j_index_threshold, "No", "Yes") %>%
           as.factor) -> opt_thresh_preds


opt_thresh_preds %>%
  metrics(truth = bid_premium, estimate = prediction) %>%
  bind_rows(
    opt_thresh_preds %>%
      sens(truth = bid_premium, estimate = prediction)
  ) %>%
  bind_rows(
    opt_thresh_preds %>%
      yardstick::spec(truth = bid_premium, estimate = prediction)
  )

#vignette("equivocal-zones", "probably")
## find max uncertain zones. when the model is certain -- use preds. otherwise -> uncertain. evaluate optimal thresholds.

### reportable rate

premium_test_pred %>%
  select(1:2,5) -> pred_tbl

segment_logistic_thresh <- pred_tbl %>%
  mutate(
    .pred = make_two_class_pred(
      estimate = .pred_No,
      levels = levels(bid_premium),
      threshold = max_j_index_threshold
    )
  )

segment_pred <- pred_tbl %>%
  mutate(
    .pred = make_two_class_pred(
      estimate = .pred_No,
      levels = levels(bid_premium),
      threshold = max_j_index_threshold,
      buffer = 0.05
    )
  )

segment_pred

segment_pred %>%
  count(.pred)

segment_pred %>%
  summarise(reportable = reportable_rate(.pred))

segment_pred %>%
  mutate(.pred_fct = as.factor(.pred)) %>%
  count(.pred, .pred_fct)


segment_logistic_thresh %>%
  mutate(.pred_fct = as.factor(.pred)) %>%
  yardstick::precision(bid_premium, .pred_fct)

segment_pred %>%
  mutate(.pred_fct = as.factor(.pred)) %>%
  yardstick::precision(bid_premium, .pred_fct)


segment_pred %>%
  mutate(.pred_fct = as.factor(.pred)) -> reporting_stats

reporting_stats %>%
  metrics(truth = bid_premium, estimate = .pred_fct) %>%
  bind_rows(
    reporting_stats %>%
      sens(truth = bid_premium, estimate = .pred_fct)
  ) %>%
  bind_rows(
    reporting_stats %>%
      yardstick::spec(truth = bid_premium, estimate = .pred_fct)
  )

###### different levels of buffer & results

buffer_zone = seq(0,0.4,0.01)
#i=1

lapply(1:length(buffer_zone), function(i){

  segment_pred <- pred_tbl %>%
    mutate(
      .pred = make_two_class_pred(
        estimate = .pred_No,
        levels = levels(bid_premium),
        threshold = max_j_index_threshold,
        buffer = buffer_zone[i]
      )
    )

  segment_pred %>%
    summarise(reportable = reportable_rate(.pred)) -> rep_rate

  segment_pred %>%
    mutate(.pred_fct = as.factor(.pred)) -> reporting_stats

  reporting_stats %>%
    metrics(truth = bid_premium, estimate = .pred_fct) %>%
    bind_rows(
      reporting_stats %>%
        sens(truth = bid_premium, estimate = .pred_fct)
    ) %>%
    bind_rows(
      reporting_stats %>%
        yardstick::spec(truth = bid_premium, estimate = .pred_fct)
    ) %>%
    select(-.estimator) %>%
    bind_rows(
      tibble(
        .metric = "reportable_rate",
        .estimate = rep_rate %>% pull(reportable)
      )
    ) %>%
    mutate(buffer = rep(buffer_zone[i]))

}) %>%
  bind_rows() -> buffer_tbl

buffer_tbl %>%
  ggplot(aes(x = buffer, y = .estimate, color = .metric, shape = .metric)) +
  geom_point(size = 4) +
  geom_line(alpha = 0.2) +
  xlab("Buffer size (+/- from optimal p-threshold)") +
  ylab("Estimate") +
  ggtitle("Accuracy metrics & reportable rates",
          "Given various equivocal zones") +
  theme_custom

buffer_tbl

chosen_buffer = 0.1

opt_pred <- pred_tbl %>%
  mutate(
    .pred = make_two_class_pred(
      estimate = .pred_No,
      levels = levels(bid_premium),
      threshold = max_j_index_threshold,
      buffer = chosen_buffer
    )
  )

opt_pred %<>%
  mutate(.pred_fct = as.factor(.pred)) %>%
  na.omit()



opt_pred %>% pull(bid_premium) -> actual

opt_pred %>% pull(.pred_fct) -> prediction

opt_pred %>%
  mutate(actual = bid_premium) -> res_tbl

return_results = function(res_tbl){

  res_tbl %>%
    conf_mat(truth = actual, estimate = .pred_fct) %>%
    pluck(1) %>%
    as_tibble()  -> conf_tbl

  TP = conf_tbl %>% filter(Prediction == "Yes" & Truth == "Yes") %>% pull(n)
  FP = conf_tbl %>% filter(Prediction == "Yes" & Truth == "No") %>% pull(n)
  TN = conf_tbl %>% filter(Prediction == "No" & Truth == "No") %>% pull(n)
  FN = conf_tbl %>% filter(Prediction == "No" & Truth == "Yes") %>% pull(n)

  spec = TN / (TN + FP)
  sens = TP / (TP + FN)
  acc = (TP + TN) / ( sum(conf_tbl %>% pull(n) ) )

  tibble(spec = spec %>% round(., 3),
         sens = sens  %>% round(., 3),
         acc = acc  %>% round(., 3)) %>% return()

}

opt_pred %>%
  mutate(actual = bid_premium)  %>%
  return_results()  -> acc_mets

paste0(acc_mets %>% colnames %>%
         paste(acc_mets %>% as.matrix() %>% as.vector, collapse = "\n"), "\n",
       paste( "Using p-threshold:",max_j_index_threshold, "and buffer zone of", chosen_buffer), collapse = "")  -> text

opt_pred %>%
  conf_mat(truth = bid_premium, estimate = .pred_fct) %>%
  pluck(1) %>%
  as_tibble() %>%
  ggplot(aes( Truth, Prediction, fill = n)) +
  geom_tile( show.legend = FALSE) +
  geom_text(aes(label = n), color = "white", alpha = 1, size = 8) +
  scale_fill_viridis_c() +
  theme_custom +
  labs(
    title = "Confusion matrix",
    subtitle = text
  ) +
  theme(panel.grid.major.x = ggplot2::element_blank(),
        panel.grid.major.y = ggplot2::element_blank())

############################################

sale_probs <- bt_model %>%
  predict(sale_testing, type = "prob") %>%
  bind_cols(sale_testing)

#sale_probs

sale_probs %>%
  gain_curve(bid_premium, .pred_No) %>%
  glimpse()


sale_probs %>%
  gain_curve(bid_premium, .pred_No) %>%
  autoplot()

sale_probs %>%
  roc_curve(bid_premium, .pred_No) %>%
  autoplot()
####

sale_probs

### optimal cut off thresholds?




################

results <-
  tibble(
    actual = sale_testing$bid_premium,
    predicted = predict(bt_model, sale_testing)$.pred_class
  ) %>%
  set_names(c("actual", "predicted"))

results$predicted#$.pred_class
conf_mat(results, truth = actual, estimate = predicted)

conf_mat(results, truth = actual, estimate = predicted)[[1]] %>%
  as_tibble() %>%
  ggplot(aes( Truth, Prediction, alpha = n)) +
  geom_tile( show.legend = FALSE) +
  geom_text(aes(label = n), color = "white", alpha = 1, size = 8) +
  theme_custom +
  labs(
    title = "Confusion matrix"
  ) +
  theme(panel.grid.major.x = ggplot2::element_blank(),
        panel.grid.major.y = ggplot2::element_blank())

predict(bt_model, sale_testing, type = "prob") %>%
  bind_cols(predict(bt_model, sale_testing)) %>%
  bind_cols(select(sale_testing, bid_premium)) %>%
  glimpse()


#######

## 10-fold cross validation
folds <- vfold_cv(data, v = 10)

folded <-
  folds %>%
  mutate(
    recipes = splits %>%
      # Prepper is a wrapper for `prep()` which handles `split` objects
      map(prepper, recipe = rec),
    test_data = splits %>% map(analysis),
    rf_fits =
      map2(
        recipes,
        test_data,
        ~ fit(
          rf_mod,
          formula(.x),
          data = bake(object = .x, newdata = .y),
          engine = "randomForest"
        )
      )
  )


## Predict
predict_rf <- function(split, rec, model) {
  test <- bake(rec, assessment(split))
  tibble(
    actual = test$diagnosis,
    predicted = predict_class(model, test)
  )
}

predictions <-
  folded %>%
  mutate(
    pred =
      list(
        splits,
        recipes,
        rf_fits
      ) %>%
      pmap(predict_rf)
  )

## Evaluate
eval <-
  predictions %>%
  mutate(
    metrics = pred %>% map(~ metrics(., truth = actual, estimate = predicted))
  ) %>%
  select(metrics) %>%
  unnest(metrics)

eval %>% knitr::kable()




