import datetime
from app.utils.logger import logger

class RetrainingPolicyEngine:
    def __init__(self, drift_threshold: float = 0.15, f1_min_threshold: float = 0.65):
        self.drift_threshold = drift_threshold
        self.f1_min_threshold = f1_min_threshold

    def evaluate_retraining_trigger(
        self,
        current_drift_score: float,
        live_f1_score: float,
        last_training_date_str: str
    ) -> dict:
        """
        Rule specification from enterprise blueprint:
        IF drift > threshold
        OR F1 drops below threshold
        OR 6 months of new data collected
        THEN retrain the model
        """
        last_train_date = datetime.datetime.strptime(last_training_date_str, "%Y-%m-%d")
        months_since_training = (datetime.datetime.now() - last_train_date).days / 30.4375

        triggers = []
        if current_drift_score > self.drift_threshold:
            triggers.append(f"Statistical drift ({current_drift_score:.3f}) exceeds threshold ({self.drift_threshold})")
            
        if live_f1_score < self.f1_min_threshold:
            triggers.append(f"Live F1-score ({live_f1_score:.3f}) dropped below threshold ({self.f1_min_threshold})")
            
        if months_since_training >= 6.0:
            triggers.append(f"Model age ({months_since_training:.1f} months) exceeds maximum 6-month lifecycle")

        should_retrain = len(triggers) > 0

        decision = {
            "retrain_recommended": should_retrain,
            "triggers_identified": triggers,
            "lifecycle_stage": "TRIGGER_RETRAINING_PIPELINE" if should_retrain else "HEALTHY_IN_PRODUCTION",
            "next_steps": [
                "1. Fetch new production ground truth labels",
                "2. Run automated validation schema checks",
                "3. Execute cross-validation on candidate models",
                "4. Compare against champion v1.0 baseline",
                "5. Promote model to models/v2 with new metadata.json"
            ] if should_retrain else ["Continue continuous prediction logging & drift monitoring"]
        }

        logger.info(f"Retraining Evaluation: {decision['lifecycle_stage']}")
        return decision

if __name__ == "__main__":
    engine = RetrainingPolicyEngine()
    result = engine.evaluate_retraining_trigger(
        current_drift_score=0.18,
        live_f1_score=0.62,
        last_training_date_str="2026-01-01"
    )
    print("=== Retraining Strategy Evaluation ===")
    for k, v in result.items():
        print(f"{k}: {v}")
