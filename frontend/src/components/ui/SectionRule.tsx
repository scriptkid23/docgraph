interface SectionRuleProps {
  thick?: boolean;
  ultra?: boolean;
}

export function SectionRule({ thick, ultra }: SectionRuleProps) {
  const className = [
    "section-rule",
    thick ? "section-rule--thick" : "",
    ultra ? "section-rule--ultra" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return <hr className={className} />;
}
