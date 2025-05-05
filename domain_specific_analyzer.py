from config import DOMAIN_TERMS, logger

class DomainSpecificAnalyzer:
    def __init__(self, domain: str = "finance"):
        self.domain = domain
        self.domain_terms = DOMAIN_TERMS.get(domain, {})

    def adapt_to_domain(self, annotation: str) -> str:
        """Адаптирует аннотацию к домену."""
        for general_term, domain_term in self.domain_terms.items():
            annotation = annotation.replace(general_term, domain_term)
        logger.info(f"Domain-adapted annotation: {annotation}")
        return annotation