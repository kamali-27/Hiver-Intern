"""
Golden Evaluation Set Builder.

Curates a balanced, realistic, and rigorous benchmark of 200 evaluation examples
combining canonical customer support queries, ambiguous inputs, hostile complaints,
explicit agent handoffs, financial refund requests, and adversarial edge cases.

Columns:
message_id, message, gold_intent, expected_response_summary, expected_decision, escalation_reason_if_any, category
"""

import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "eval", "golden_eval_set.csv")

# Curated benchmark specifications across 8 categories and edge cases
GOLDEN_DATA = [
    # -------------------------------------------------------------
    # 1. ORDER & DELIVERY DELAYS (Canonical -> AUTO_HANDLE)
    # -------------------------------------------------------------
    ("Where is my package? Order #112-9281726 was supposed to arrive yesterday!", "order_delivery_delay", "Apologize for delay, check tracking link, offer DM support", "AUTO_HANDLE", "", "canonical"),
    ("My delivery for order #402-8827162 has been delayed for 2 days. When is it coming?", "order_delivery_delay", "Apologize and provide tracking guidance", "AUTO_HANDLE", "", "canonical"),
    ("Tracking says delivered to porch for #109-2281928, but nothing is here.", "order_delivery_delay", "Advise checking around property and DMing details", "AUTO_HANDLE", "", "canonical"),
    ("Package #551-8291029 has been stuck at the transit facility for 4 days.", "order_delivery_delay", "Check carrier transit status via DM", "AUTO_HANDLE", "", "canonical"),
    ("Paid for Prime 1-day shipping on order #223-9182736 and it hasn't even shipped yet.", "order_delivery_delay", "Apologize for expedited shipping delay and review order", "AUTO_HANDLE", "", "canonical"),
    ("Driver marked delivery attempted for #771-2918273, but I was home all day.", "order_delivery_delay", "Advise redelivery window and offer support", "AUTO_HANDLE", "", "canonical"),
    ("Can you give me an update on package #334-1182938? It's a birthday present.", "order_delivery_delay", "Check status and provide assistance via DM", "AUTO_HANDLE", "", "canonical"),
    ("Why is my package taking a week longer than promised for order #992-1827361?", "order_delivery_delay", "Apologize and review logistics delay", "AUTO_HANDLE", "", "canonical"),
    ("My item was shipped via USPS for order #114-8837192, tracking not updating.", "order_delivery_delay", "Advise carrier update lag and check order", "AUTO_HANDLE", "", "canonical"),
    ("Package #552-1928374 shows out for delivery since 8am, still not here at 9pm.", "order_delivery_delay", "Explain delivery hours and offer assistance", "AUTO_HANDLE", "", "canonical"),
    ("Order #331-9872615 hasn't moved from the origin depot in 72 hours.", "order_delivery_delay", "Investigate depot hold via DM", "AUTO_HANDLE", "", "canonical"),
    ("Carrier left order #662-1829374 in the rain, box is soaked.", "order_delivery_delay", "Apologize for poor delivery handling and review damage", "AUTO_HANDLE", "", "canonical"),
    ("Was supposed to receive order #441-2918273 before noon today. Nothing yet.", "order_delivery_delay", "Check scheduled delivery appointment", "AUTO_HANDLE", "", "canonical"),
    ("Tracking number shows delivered to wrong city for order #882-1928371!", "order_delivery_delay", "Investigate misrouted shipment immediately", "AUTO_HANDLE", "", "canonical"),
    ("Any update on shipment #991-2819283? Order placed 5 days ago.", "order_delivery_delay", "Provide shipping status and carrier link", "AUTO_HANDLE", "", "canonical"),
    ("Order #118-2918273 has no tracking updates for the past 48 hours.", "order_delivery_delay", "Check tracking status via DM", "AUTO_HANDLE", "", "canonical"),
    ("My package #772-1928372 was returned to sender, why?", "order_delivery_delay", "Investigate return to sender reason", "AUTO_HANDLE", "", "canonical"),
    ("Delivery window missed for order #553-9182736. Can I reschedule?", "order_delivery_delay", "Explain rescheduling options", "AUTO_HANDLE", "", "canonical"),
    ("Package #442-1829381 says handed to resident, but nobody was home!", "order_delivery_delay", "Check POD details and DM customer", "AUTO_HANDLE", "", "canonical"),
    ("Is order #881-2918273 arriving today as scheduled?", "order_delivery_delay", "Confirm delivery date via tracking", "AUTO_HANDLE", "", "canonical"),

    # -------------------------------------------------------------
    # 2. REFUND REQUESTS (High-Risk -> ESCALATE_TO_HUMAN)
    # -------------------------------------------------------------
    ("I returned order #102-3921827 a week ago and haven't received my refund of $85.00.", "refund_request", "Explain refund timeline and verify return receipt", "ESCALATE_TO_HUMAN", "Financial refund request requiring specialist verification", "canonical"),
    ("Please cancel order #332-1928371 and process a full refund right now.", "refund_request", "Process cancellation and refund request", "ESCALATE_TO_HUMAN", "Financial refund request requiring specialist verification", "canonical"),
    ("Item arrived shattered in order #551-2918273. I want a complete refund, not a replacement.", "refund_request", "Acknowledge damaged item and process refund", "ESCALATE_TO_HUMAN", "Financial refund request requiring specialist verification", "canonical"),
    ("My refund of $120.00 for order #882-1928374 is 8 days overdue. Where is my money?", "refund_request", "Investigate overdue refund transaction", "ESCALATE_TO_HUMAN", "Financial refund request requiring specialist verification", "canonical"),
    ("Drop-off return locker accepted my item 6 days ago for #441-2918273, no refund email yet.", "refund_request", "Verify drop-off locker scan and issue refund", "ESCALATE_TO_HUMAN", "Financial refund request requiring specialist verification", "canonical"),
    ("Need a refund on order #771-2918274. The seller sent the wrong shoe size.", "refund_request", "Assist with return label and refund", "ESCALATE_TO_HUMAN", "Financial refund request requiring specialist verification", "canonical"),
    ("Why was my refund for order #662-1928371 credited as a gift card instead of my card?", "refund_request", "Review refund payment method", "ESCALATE_TO_HUMAN", "Financial refund request requiring specialist verification", "canonical"),
    ("Returned order #992-1827362 on Monday. Please confirm refund status.", "refund_request", "Check return inspection and refund release", "ESCALATE_TO_HUMAN", "Financial refund request requiring specialist verification", "canonical"),
    ("Cancel my subscription order #113-2918273 and reverse the $35 charge.", "refund_request", "Process subscription refund", "ESCALATE_TO_HUMAN", "Financial refund request requiring specialist verification", "canonical"),
    ("I was promised a promotional refund of $20 on order #442-1928371 that never posted.", "refund_request", "Apply promised promotional credit", "ESCALATE_TO_HUMAN", "Financial refund request requiring specialist verification", "canonical"),
    ("Return tracking shows item delivered back to warehouse for #552-1928371. Refund please.", "refund_request", "Verify warehouse receipt and release refund", "ESCALATE_TO_HUMAN", "Financial refund request requiring specialist verification", "canonical"),
    ("Can I get a refund for order #331-2918273? The food item expired before delivery.", "refund_request", "Process refund for perishable goods", "ESCALATE_TO_HUMAN", "Financial refund request requiring specialist verification", "canonical"),
    ("I want to return order #881-2918274 and get a full refund to my PayPal account.", "refund_request", "Explain return steps and refund mechanism", "ESCALATE_TO_HUMAN", "Financial refund request requiring specialist verification", "canonical"),
    ("Charged restocking fee on return for order #221-1928371. I want that refunded.", "refund_request", "Review restocking fee dispute", "ESCALATE_TO_HUMAN", "Financial refund request requiring specialist verification", "canonical"),
    ("Where is my refund for return label 1Z99281726? Shipped 10 days ago.", "refund_request", "Track return shipment and verify refund", "ESCALATE_TO_HUMAN", "Financial refund request requiring specialist verification", "canonical"),
    ("Order #773-1928371 was stolen from building lobby, need a full refund.", "refund_request", "Assist with stolen package claim and refund", "ESCALATE_TO_HUMAN", "Financial refund request requiring specialist verification", "canonical"),
    ("How long do credit card refunds usually take after item inspection for #443-1928371?", "refund_request", "Explain 3-5 day banking cycle", "ESCALATE_TO_HUMAN", "Financial refund request requiring specialist verification", "canonical"),
    ("I sent back two items in one box for order #883-1928371, only got refunded for one.", "refund_request", "Investigate partial return discrepancy", "ESCALATE_TO_HUMAN", "Financial refund request requiring specialist verification", "canonical"),
    ("Please expedite refund for order #993-1827361, need funds urgently.", "refund_request", "Expedite refund review with billing team", "ESCALATE_TO_HUMAN", "Financial refund request requiring specialist verification", "canonical"),
    ("Item missing from return batch for order #115-2918273. Help with refund.", "refund_request", "Review return contents and resolve refund", "ESCALATE_TO_HUMAN", "Financial refund request requiring specialist verification", "canonical"),

    # -------------------------------------------------------------
    # 3. ACCOUNT ACCESS & SECURITY (AUTO_HANDLE - security guidance)
    # -------------------------------------------------------------
    ("Locked out of my account. OTP code is not arriving on my phone.", "account_access", "Provide 2FA reset and security link guidance", "AUTO_HANDLE", "", "canonical"),
    ("Forgot my password and reset email link never shows up in spam or inbox.", "account_access", "Guide customer through account recovery flow", "AUTO_HANDLE", "", "canonical"),
    ("My account was hacked and someone ordered gift cards! Please lock it!", "account_access", "Urgent security lockdown steps and DM review", "ESCALATE_TO_HUMAN", "Hostile/urgent security compromise detected", "edge_case"),
    ("Cannot log in from my new tablet, says suspicious device detected.", "account_access", "Provide device authentication instructions", "AUTO_HANDLE", "", "canonical"),
    ("Changed my phone number and now cannot pass two-step verification.", "account_access", "Provide 2FA recovery options", "AUTO_HANDLE", "", "canonical"),
    ("Login page keeps looping back to enter email without showing password box.", "account_access", "Troubleshoot browser cache and cookies", "AUTO_HANDLE", "", "canonical"),
    ("How do I update my primary email address on my Amazon profile?", "account_access", "Explain account profile settings steps", "AUTO_HANDLE", "", "canonical"),
    ("My account has been temporarily on hold due to unusual activity.", "account_access", "Provide account review documentation link", "AUTO_HANDLE", "", "canonical"),
    ("Authenticator app codes are showing as invalid when logging in.", "account_access", "Provide time-sync and backup code guidance", "AUTO_HANDLE", "", "canonical"),
    ("Someone unauthorized accessed my Prime Video profile.", "account_access", "Provide PIN reset and device de-registration guidance", "AUTO_HANDLE", "", "canonical"),
    ("Cannot sign in to Amazon app on iPhone after latest iOS update.", "account_access", "Advise app reinstall and credential verification", "AUTO_HANDLE", "", "canonical"),
    ("Account locked for entering incorrect password too many times.", "account_access", "Explain unlock cooldown and recovery options", "AUTO_HANDLE", "", "canonical"),
    ("How do I enable two-factor authentication on my account?", "account_access", "Provide 2FA setup link and steps", "AUTO_HANDLE", "", "canonical"),
    ("Keep getting unexpected password reset notification emails.", "account_access", "Provide security checkup recommendations", "AUTO_HANDLE", "", "canonical"),
    ("Unable to log in to Amazon seller central with customer login.", "account_access", "Direct to seller central auth portal", "AUTO_HANDLE", "", "canonical"),
    ("Password reset link says token expired immediately upon clicking.", "account_access", "Advise requesting fresh token in private window", "AUTO_HANDLE", "", "canonical"),
    ("Can I merge two separate Amazon accounts registered under different emails?", "account_access", "Explain account merging policy via Household", "AUTO_HANDLE", "", "canonical"),
    ("SMS verification code delayed by 30 minutes, by then it's expired.", "account_access", "Suggest authenticator app alternative", "AUTO_HANDLE", "", "canonical"),
    ("Lost my phone with authenticator app. How can I log into my account?", "account_access", "Direct to government ID verification recovery", "AUTO_HANDLE", "", "canonical"),
    ("Error saying account suspended when trying to browse digital orders.", "account_access", "Direct to account specialist review", "AUTO_HANDLE", "", "canonical"),

    # -------------------------------------------------------------
    # 4. BILLING & SUBSCRIPTION ISSUES (AUTO_HANDLE / ESCALATE)
    # -------------------------------------------------------------
    ("I was charged $14.99 twice for order #112-9827162. Please check duplicate charge.", "billing_issue", "Explain authorization hold and investigate double charge", "AUTO_HANDLE", "", "canonical"),
    ("Why was my credit card charged $139 for Prime when I never signed up?", "billing_issue", "Investigate unauthorized Prime membership fee", "AUTO_HANDLE", "", "canonical"),
    ("Invoice shows an unexpected digital service fee of $4.99 on order #332-1827361.", "billing_issue", "Explain digital order fee breakdown", "AUTO_HANDLE", "", "canonical"),
    ("My payment was declined for order #551-2918274 but bank says funds are fine.", "billing_issue", "Advise updating payment method details", "AUTO_HANDLE", "", "canonical"),
    ("Turned off auto-renewal last month, but was still billed for Music Unlimited.", "billing_issue", "Check subscription status and reverse fee", "AUTO_HANDLE", "", "canonical"),
    ("Can you send me a formal VAT invoice for my business order #882-1928371?", "billing_issue", "Provide VAT invoice download instructions", "AUTO_HANDLE", "", "canonical"),
    ("Charged $79.00 on my debit card with description AMZN Mktp, cannot find order.", "billing_issue", "Help identify transaction across family accounts", "AUTO_HANDLE", "", "canonical"),
    ("Why does my bank statement show 3 separate charges for 1 order?", "billing_issue", "Explain split shipments and per-shipment billing", "AUTO_HANDLE", "", "canonical"),
    ("Payment failed on subscription renewal, how do I retry with different card?", "billing_issue", "Guide through payment wallet settings", "AUTO_HANDLE", "", "canonical"),
    ("Charged $29.99 for Kindle Unlimited that I cancelled during free trial.", "billing_issue", "Review trial cancellation and refund charge", "AUTO_HANDLE", "", "canonical"),
    ("Double billed on my Visa card for order #441-2918273. Fix this!", "billing_issue", "Review duplicate billing via DM", "AUTO_HANDLE", "", "canonical"),
    ("Unexpected recurring charge of $9.99 appearing on my credit card every month.", "billing_issue", "Help locate active channels or subscriptions", "AUTO_HANDLE", "", "canonical"),
    ("Why was my promotional credit not applied at checkout for order #771-2918273?", "billing_issue", "Verify promo terms and apply balance", "AUTO_HANDLE", "", "canonical"),
    ("Gift card balance was deducted but order was cancelled, where is my balance?", "billing_issue", "Explain immediate gift card balance reinstatement", "AUTO_HANDLE", "", "canonical"),
    ("Charged international foreign transaction fee on a domestic order.", "billing_issue", "Explain bank currency conversion policies", "AUTO_HANDLE", "", "canonical"),
    ("Need itemized receipt for tax reimbursement on order #992-1827361.", "billing_issue", "Direct to printable invoice section", "AUTO_HANDLE", "", "canonical"),
    ("Bank declined authorization for #221-1928374, how long do I have to update card?", "billing_issue", "Provide 24-48h grace period details", "AUTO_HANDLE", "", "canonical"),
    ("Why did Prime annual membership price increase without prior notice?", "billing_issue", "Explain membership terms and price update schedule", "AUTO_HANDLE", "", "canonical"),
    ("Charged twice for one rental movie on Prime Video.", "billing_issue", "Reverse duplicate video rental charge", "AUTO_HANDLE", "", "canonical"),
    ("Payment status stuck in processing for over 12 hours for order #661-1928371.", "billing_issue", "Investigate payment gateway status", "AUTO_HANDLE", "", "canonical"),

    # -------------------------------------------------------------
    # 5. TECHNICAL BUGS & APP GLITCHES (AUTO_HANDLE)
    # -------------------------------------------------------------
    ("Your iOS app keeps crashing whenever I click the place order button. iOS 18.", "technical_bug", "Troubleshoot app cache, update, and reinstall", "AUTO_HANDLE", "", "canonical"),
    ("Website returns error 500 internal server error when opening cart page.", "technical_bug", "Advise clearing browser cookies and incognito mode", "AUTO_HANDLE", "", "canonical"),
    ("Search bar on desktop Chrome is completely frozen and won't accept input.", "technical_bug", "Advise browser extension check and reload", "AUTO_HANDLE", "", "canonical"),
    ("Images not loading on product listings, just seeing empty gray boxes.", "technical_bug", "Advise checking connection and app cache", "AUTO_HANDLE", "", "canonical"),
    ("Getting 'An unexpected error occurred' when saving new shipping address.", "technical_bug", "Provide address formatting check and retry advice", "AUTO_HANDLE", "", "canonical"),
    ("Prime Video app on Samsung Smart TV keeps buffering every 30 seconds.", "technical_bug", "Troubleshoot TV app cache and internet connection", "AUTO_HANDLE", "", "canonical"),
    ("Wishlist won't let me delete saved items, throws error updating list.", "technical_bug", "Advise web portal workaround and report bug", "AUTO_HANDLE", "", "canonical"),
    ("One-click checkout button is missing from product pages in browser.", "technical_bug", "Check 1-Click purchasing settings", "AUTO_HANDLE", "", "canonical"),
    ("Kindle app won't sync reading progress between iPad and phone.", "technical_bug", "Guide Whispersync settings troubleshooting", "AUTO_HANDLE", "", "canonical"),
    ("Review submission form is broken, submit button remains grayed out.", "technical_bug", "Check character requirements and browser compatibility", "AUTO_HANDLE", "", "canonical"),
    ("Notification settings in Android app keep resetting to default.", "technical_bug", "Advise OS notification permissions check", "AUTO_HANDLE", "", "canonical"),
    ("Checkout page freezes when selecting saved debit card payment.", "technical_bug", "Advise clearing saved payment session", "AUTO_HANDLE", "", "canonical"),
    ("Filter by Prime delivery option is broken in search results.", "technical_bug", "Report search filter glitch to tech team", "AUTO_HANDLE", "", "canonical"),
    ("Audio and video out of sync on Prime Video browser playback.", "technical_bug", "Advise hardware acceleration toggle in Chrome", "AUTO_HANDLE", "", "canonical"),
    ("App crashes instantly upon launch on Pixel 9 running Android 15.", "technical_bug", "Request OS version and advise update check", "AUTO_HANDLE", "", "canonical"),
    ("Add to Cart button does nothing on product B09V3K1M4P.", "technical_bug", "Check seller inventory status and browser cache", "AUTO_HANDLE", "", "canonical"),
    ("Digital book download failed with license limit reached error.", "technical_bug", "Guide managing digital content devices", "AUTO_HANDLE", "", "canonical"),
    ("Barcode scanner inside mobile app won't open camera.", "technical_bug", "Guide camera permission toggle in device settings", "AUTO_HANDLE", "", "canonical"),
    ("Order history page pagination button to page 2 doesn't load.", "technical_bug", "Advise checking year filter in order history", "AUTO_HANDLE", "", "canonical"),
    ("Website stylesheet seems broken, everything showing as unformatted text.", "technical_bug", "Advise hard reload (Ctrl+F5) to clear CDN cache", "AUTO_HANDLE", "", "canonical"),

    # -------------------------------------------------------------
    # 6. GENERAL INQUIRIES & POLICIES (AUTO_HANDLE)
    # -------------------------------------------------------------
    ("Do you ship electronics internationally to France and what are the fees?", "general_inquiry", "Provide international shipping policy link", "AUTO_HANDLE", "", "canonical"),
    ("What is the return window for items bought during Black Friday?", "general_inquiry", "Explain extended holiday return policy", "AUTO_HANDLE", "", "canonical"),
    ("Can I use an Amazon gift card to pay for Prime video rentals?", "general_inquiry", "Confirm gift card eligibility for digital purchases", "AUTO_HANDLE", "", "canonical"),
    ("Do you offer student discounts on annual Prime memberships?", "general_inquiry", "Provide Prime Student discount information", "AUTO_HANDLE", "", "canonical"),
    ("Is manufacturer warranty valid when buying through third party sellers?", "general_inquiry", "Explain authorized seller warranty policies", "AUTO_HANDLE", "", "canonical"),
    ("How do I change the delivery address on an order that hasn't shipped yet?", "general_inquiry", "Guide order modification before dispatch", "AUTO_HANDLE", "", "canonical"),
    ("Are your delivery packaging boxes recyclable?", "general_inquiry", "Provide Amazon Second Chance recycling info", "AUTO_HANDLE", "", "canonical"),
    ("What are the customer service operating hours on weekends?", "general_inquiry", "State 24/7 customer care availability", "AUTO_HANDLE", "", "canonical"),
    ("Does this wireless mouse come with batteries included in the box?", "general_inquiry", "Advise checking package contents on listing", "AUTO_HANDLE", "", "canonical"),
    ("How can I purchase an Amazon e-gift card for a friend?", "general_inquiry", "Direct to gift card digital store", "AUTO_HANDLE", "", "canonical"),
    ("Do you price match with Best Buy or Walmart?", "general_inquiry", "Explain competitive pricing policy", "AUTO_HANDLE", "", "canonical"),
    ("Can I pick up my package from an Amazon Hub locker near my house?", "general_inquiry", "Explain Hub Locker delivery selection at checkout", "AUTO_HANDLE", "", "canonical"),
    ("What happens if a package requires signature and I am not home?", "general_inquiry", "Explain carrier redelivery attempt policy", "AUTO_HANDLE", "", "canonical"),
    ("How do I cancel my Prime membership before renewal date?", "general_inquiry", "Direct to Manage Prime Membership portal", "AUTO_HANDLE", "", "canonical"),
    ("Can two family members share one Prime shipping benefit?", "general_inquiry", "Explain Amazon Household sharing rules", "AUTO_HANDLE", "", "canonical"),
    ("What is the maximum weight limit for Amazon Locker delivery?", "general_inquiry", "Provide locker dimensions and weight specs", "AUTO_HANDLE", "", "canonical"),
    ("Does Amazon offer trade-in credit for old Kindle devices?", "general_inquiry", "Direct to Amazon Trade-In program", "AUTO_HANDLE", "", "canonical"),
    ("How do I report a missing item from a multi-item package?", "general_inquiry", "Guide customer through Your Orders missing item report", "AUTO_HANDLE", "", "canonical"),
    ("Can I pay using two different credit cards on one order?", "general_inquiry", "Explain payment splitting rules (Gift card + CC)", "AUTO_HANDLE", "", "canonical"),
    ("What are the perks of Amazon Prime for Kindle book readers?", "general_inquiry", "Explain Prime Reading benefits", "AUTO_HANDLE", "", "canonical"),

    # -------------------------------------------------------------
    # 7. COMPLAINT ESCALATION (High-Risk -> ESCALATE_TO_HUMAN)
    # -------------------------------------------------------------
    ("WORST CUSTOMER SERVICE EVER! Agent hung up on me after 40 minutes on hold!", "complaint_escalation", "Apologize sincerely and route to senior escalation team", "ESCALATE_TO_HUMAN", "Severe customer dissatisfaction requires human empathy and resolution.", "canonical"),
    ("I demand to speak to a supervisor or operations manager immediately.", "complaint_escalation", "Acknowledge escalation request and arrange manager contact", "ESCALATE_TO_HUMAN", "Explicit request for supervisor/manager", "canonical"),
    ("This is an outright scam! You took my money and won't send my product!", "complaint_escalation", "De-escalate fraud allegation and route to specialist", "ESCALATE_TO_HUMAN", "Hostile sentiment and fraud allegations", "canonical"),
    ("I will be contacting my attorney and filing a lawsuit over order #112-9821726.", "complaint_escalation", "De-escalate legal threat and route to executive relations", "ESCALATE_TO_HUMAN", "Legal action threat detected", "canonical"),
    ("Reporting your company to the Better Business Bureau for deceptive practices.", "complaint_escalation", "Route regulatory complaint to senior leadership", "ESCALATE_TO_HUMAN", "Regulatory/BBB complaint threat", "canonical"),
    ("Your customer support rep was rude and insulted me in chat. Unacceptable!", "complaint_escalation", "Sincere apology and escalate rep misconduct to manager", "ESCALATE_TO_HUMAN", "Agent misconduct complaint", "canonical"),
    ("I've been a Prime customer for 12 years and never been treated so disgracefully.", "complaint_escalation", "Acknowledge long-term loyalty and offer senior review", "ESCALATE_TO_HUMAN", "Severe customer dissatisfaction", "canonical"),
    ("Your negligence ruined my daughter's birthday party. Disgusting service.", "complaint_escalation", "Express deep empathy and route to executive team", "ESCALATE_TO_HUMAN", "Severe dissatisfaction and distress", "canonical"),
    ("Transfer me to a real human being right now. Stop sending automated garbage!", "complaint_escalation", "Immediately route to live human support specialist", "ESCALATE_TO_HUMAN", "Explicit human agent request", "canonical"),
    ("You guys stole $400 from my checking account! Absolute thieves!", "complaint_escalation", "Address theft accusation and route to fraud unit", "ESCALATE_TO_HUMAN", "Theft allegations and hostile sentiment", "canonical"),
    ("I have called 6 times this week about order #551-2918274 and got 6 different lies.", "complaint_escalation", "Acknowledge broken promises and assign single case owner", "ESCALATE_TO_HUMAN", "Repeated failure and customer frustration", "canonical"),
    ("Give me the corporate email for Jeff Bezos or the CEO right now.", "complaint_escalation", "Route executive escalation request to leadership team", "ESCALATE_TO_HUMAN", "Executive escalation request", "canonical"),
    ("Your driver threw my fragile package over an 8-foot fence and broke it!", "complaint_escalation", "Apologize for carrier property damage and escalate claim", "ESCALATE_TO_HUMAN", "Carrier misconduct and property damage", "canonical"),
    ("This is completely unacceptable negligence. I expect a call from a manager today.", "complaint_escalation", "Schedule manager callback", "ESCALATE_TO_HUMAN", "Manager escalation request", "canonical"),
    ("Filing a police report for stolen package order #332-1928371.", "complaint_escalation", "Assist with police report reference and claims unit", "ESCALATE_TO_HUMAN", "Police/legal action threat", "canonical"),
    ("I want an explanation from someone who actually has authority to fix this.", "complaint_escalation", "Connect customer with authorized lead", "ESCALATE_TO_HUMAN", "Request for supervisory authority", "canonical"),
    ("Your automated bot is wasting my time. Connect me to a person!", "complaint_escalation", "Seamless handoff to human representative", "ESCALATE_TO_HUMAN", "Explicit human agent request", "canonical"),
    ("Terrible delivery, broken promise, and zero accountability. Shame on you!", "complaint_escalation", "Apologize and review service failure with supervisor", "ESCALATE_TO_HUMAN", "Severe customer dissatisfaction", "canonical"),
    ("I will publicize this disgusting chat on social media and tag tech news.", "complaint_escalation", "De-escalate public relations risk with priority team", "ESCALATE_TO_HUMAN", "Public social media complaint threat", "canonical"),
    ("Can I speak to someone who actually understands English and can read notes?", "complaint_escalation", "Politely connect to human specialist", "ESCALATE_TO_HUMAN", "Explicit human handoff and agent frustration", "canonical"),

    # -------------------------------------------------------------
    # 8. POSITIVE FEEDBACK (AUTO_HANDLE)
    # -------------------------------------------------------------
    ("Thank you so much Sarah from support who resolved my delivery issue in 3 minutes! 5 stars!", "positive_feedback", "Express gratitude and share compliments with Sarah", "AUTO_HANDLE", "", "canonical"),
    ("Just wanted to say thank you for the speedy refund on order #112-9827162. Amazing service!", "positive_feedback", "Thank customer and wish them a wonderful day", "AUTO_HANDLE", "", "canonical"),
    ("Shoutout to your team for delivering my package early on a holiday! Kudos!", "positive_feedback", "Celebrate positive experience and thank customer", "AUTO_HANDLE", "", "canonical"),
    ("Incredible customer service today. Truly grateful for the prompt help!", "positive_feedback", "Acknowledge appreciation and customer loyalty", "AUTO_HANDLE", "", "canonical"),
    ("Your support agent John was courteous, patient, and solved my account issue.", "positive_feedback", "Pass praise to agent John and team leadership", "AUTO_HANDLE", "", "canonical"),
    ("Best online shopping customer support experience I've had in years. Thank you!", "positive_feedback", "Express heartfelt gratitude for feedback", "AUTO_HANDLE", "", "canonical"),
    ("Replaced my defective coffee maker with zero hassle. Love you guys!", "positive_feedback", "Express delight that replacement was smooth", "AUTO_HANDLE", "", "canonical"),
    ("5 stars to Amazon support for always having my back when things go wrong!", "positive_feedback", "Thank customer for their continued trust", "AUTO_HANDLE", "", "canonical"),
    ("Super fast resolution this morning. You guys are the gold standard!", "positive_feedback", "Thank customer warmly for the compliment", "AUTO_HANDLE", "", "canonical"),
    ("Thank you @AmazonHelp for helping me track down my grandmother's birthday gift!", "positive_feedback", "Share joy in resolving birthday gift delivery", "AUTO_HANDLE", "", "canonical"),
    ("Customer support handled my return seamlessly. Very happy customer here!", "positive_feedback", "Thank customer and reinforce return ease", "AUTO_HANDLE", "", "canonical"),
    ("Big thanks to the team for fixing my Prime Video login glitch so fast.", "positive_feedback", "Express gratitude and glad tech issue is resolved", "AUTO_HANDLE", "", "canonical"),
    ("Courteous, helpful, and professional service today. Thank you so much!", "positive_feedback", "Acknowledge kind words warmly", "AUTO_HANDLE", "", "canonical"),
    ("Amazon customer service is unbeatable. Thanks for the quick replacement!", "positive_feedback", "Thank customer for positive endorsement", "AUTO_HANDLE", "", "canonical"),
    ("Appreciate the friendly help with my invoice query this morning!", "positive_feedback", "Thank customer and remain available for help", "AUTO_HANDLE", "", "canonical"),
    ("Prompt, polite, and effective support. Thank you Amazon team!", "positive_feedback", "Warmly acknowledge and thank customer", "AUTO_HANDLE", "", "canonical"),
    ("Thank you for resolving the delivery delay on #441-2918273 ahead of time.", "positive_feedback", "Express pleasure in meeting delivery needs", "AUTO_HANDLE", "", "canonical"),
    ("Great customer service experience, solved my problem on the first contact.", "positive_feedback", "Celebrate first-contact resolution success", "AUTO_HANDLE", "", "canonical"),
    ("Huge thanks to the representative who guided me through 2FA setup.", "positive_feedback", "Acknowledge helpfulness and pass praise", "AUTO_HANDLE", "", "canonical"),
    ("Just dropping by to say you guys do awesome work every day. Thanks!", "positive_feedback", "Warm appreciation for unsolicited kindness", "AUTO_HANDLE", "", "canonical"),

    # -------------------------------------------------------------
    # 9. ADVERSARIAL EDGE CASES & SPECIAL TEST SCENARIOS
    # -------------------------------------------------------------
    # Multi-intent / Ambiguous queries
    ("I want to return order #102-3921827 and also my app crashed while browsing.", "refund_request", "Handle return request and note app stability troubleshooting", "ESCALATE_TO_HUMAN", "Financial refund request combined with technical bug", "multi_intent"),
    ("Delayed package #441-2918273 and your support agent was completely useless.", "complaint_escalation", "Apologize for both delay and poor service, escalate to supervisor", "ESCALATE_TO_HUMAN", "Severe complaint and agent dissatisfaction", "multi_intent"),
    ("Where is my refund for the order that was never delivered in the first place?", "refund_request", "Investigate undelivered shipment and process refund", "ESCALATE_TO_HUMAN", "Financial refund request", "multi_intent"),
    ("Did you charge me twice because the website bugged out during checkout?", "billing_issue", "Verify billing transaction against web glitch", "AUTO_HANDLE", "", "multi_intent"),

    # Typos and extreme colloquialisms
    ("wher is my ordr pkg late 3 days", "order_delivery_delay", "Apologize for delay and request order number to track", "AUTO_HANDLE", "", "typos_slang"),
    ("plz rfund my mony for brokn itm", "refund_request", "Help with damaged item refund steps", "ESCALATE_TO_HUMAN", "Financial refund request", "typos_slang"),
    ("cant log in otp code not workin", "account_access", "Provide 2FA reset and security guidance", "AUTO_HANDLE", "", "typos_slang"),
    ("ap chrash evry tiem i clik bay", "technical_bug", "Troubleshoot app crash on purchase", "AUTO_HANDLE", "", "typos_slang"),
    ("thx u guys r awesome <3", "positive_feedback", "Thank customer warmly for appreciation", "AUTO_HANDLE", "", "typos_slang"),
    ("u charge my card 2 times wtf", "billing_issue", "De-escalate and investigate duplicate charge", "AUTO_HANDLE", "", "typos_slang"),

    # Non-English / Multilingual fragments
    ("Mon colis n'est toujours pas arrivé, commande #112-9982716.", "order_delivery_delay", "Respond or route to localized French customer support", "AUTO_HANDLE", "", "non_english"),
    ("Donde esta mi paquete por favor? Orden #331-2918273.", "order_delivery_delay", "Respond or route to localized Spanish support", "AUTO_HANDLE", "", "non_english"),
    ("Wo bleibt meine Rückerstattung für Bestellung #551-2918274?", "refund_request", "Provide refund information in German / DM assist", "ESCALATE_TO_HUMAN", "Financial refund request", "non_english"),
    ("Il mio account è bloccato, aiuto per favore.", "account_access", "Provide Italian account recovery guidance", "AUTO_HANDLE", "", "non_english"),

    # Short fragments, borderline gibberish, empty-like inputs
    ("help", "general_inquiry", "Politely inquire how support can assist customer", "ESCALATE_TO_HUMAN", "Ambiguous short fragment below confidence threshold", "fragment"),
    ("???", "general_inquiry", "Politely ask how we can help", "ESCALATE_TO_HUMAN", "Low similarity and confidence punctuation fragment", "fragment"),
    ("asdkfjalskdfj ???", "general_inquiry", "Politely ask customer to rephrase query", "ESCALATE_TO_HUMAN", "Gibberish input with no historical match", "fragment"),
    ("order", "order_delivery_delay", "Ask for order number and specific inquiry", "ESCALATE_TO_HUMAN", "Extremely underspecified input", "fragment"),
    ("cancel", "refund_request", "Inquire which order customer wishes to cancel", "ESCALATE_TO_HUMAN", "Financial cancellation request", "fragment"),
    ("human representative please", "complaint_escalation", "Directly route to human customer representative", "ESCALATE_TO_HUMAN", "Explicit human request", "explicit_human"),
    ("agent agent agent", "complaint_escalation", "Directly route to customer service agent", "ESCALATE_TO_HUMAN", "Explicit human request", "explicit_human"),
    ("speak to representative now", "complaint_escalation", "Connect to live support representative", "ESCALATE_TO_HUMAN", "Explicit human request", "explicit_human"),
    ("Can a human being read this conversation?", "complaint_escalation", "Confirm human handover and review notes", "ESCALATE_TO_HUMAN", "Explicit human request", "explicit_human"),
    ("I need to speak to your manager immediately.", "complaint_escalation", "Escalate case to customer support supervisor", "ESCALATE_TO_HUMAN", "Explicit supervisor request", "explicit_human"),

    # High-value customer dispute edge cases
    ("My $2,500 laptop arrived with an empty box! Delivery driver stole it!", "complaint_escalation", "Urgent high-value stolen goods investigation", "ESCALATE_TO_HUMAN", "High-value theft allegation requiring senior investigation", "edge_case"),
    ("You charged my corporate card $12,000 for cloud servers without approval.", "billing_issue", "Route enterprise billing discrepancy to specialist", "AUTO_HANDLE", "", "edge_case"),
    ("The medication in order #992-1827361 was delivered spoiled and ruined.", "complaint_escalation", "Urgent perishable health item escalation", "ESCALATE_TO_HUMAN", "Perishable health item delivery failure", "edge_case"),
    ("Defective phone charger caught on fire and melted my nightstand.", "complaint_escalation", "Safety defect incident escalated to executive product safety team", "ESCALATE_TO_HUMAN", "Product safety hazard and property damage", "edge_case"),
    ("My child accidentally purchased $500 of in-game items on Prime tablet.", "refund_request", "Assist parent with parental controls and refund review", "ESCALATE_TO_HUMAN", "Financial refund dispute for unauthorized minor purchase", "edge_case"),
    ("The delivery driver assaulted my dog in the front yard!", "complaint_escalation", "Urgent driver safety incident routed to executive dispatch", "ESCALATE_TO_HUMAN", "Severe physical safety incident", "edge_case"),
    ("Identity theft: someone opened 4 credit cards in my name on Amazon.", "account_access", "Urgent fraud and identity security team handoff", "ESCALATE_TO_HUMAN", "Severe identity theft and fraud", "edge_case"),
    ("Order delivered to a different state entirely according to FedEx tracking.", "order_delivery_delay", "Investigate severe logistics misroute", "AUTO_HANDLE", "", "edge_case"),
    ("Received customer support email with another customer's personal data.", "complaint_escalation", "Report data privacy incident to compliance office", "ESCALATE_TO_HUMAN", "Privacy and data breach report", "edge_case"),
    ("Can I get a discount code for my wedding registry?", "general_inquiry", "Provide registry completion discount details", "AUTO_HANDLE", "", "edge_case")
]

def build_golden_set():
    """Builds and writes golden_eval_set.csv."""
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    rows = []
    for i, item in enumerate(GOLDEN_DATA, 1):
        msg, intent, exp_summary, decision, esc_reason, category = item
        rows.append({
            "message_id": f"GOLD_{i:03d}",
            "message": msg,
            "gold_intent": intent,
            "expected_response_summary": exp_summary,
            "expected_decision": decision,
            "escalation_reason_if_any": esc_reason,
            "category": category
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"[build_golden_set] Successfully created {len(df)} golden evaluation examples at {OUTPUT_PATH}")
    print("\nIntent Breakdown in Golden Set:")
    print(df["gold_intent"].value_counts().to_string())
    print("\nDecision Breakdown in Golden Set:")
    print(df["expected_decision"].value_counts().to_string())
    print("\nCategory Breakdown:")
    print(df["category"].value_counts().to_string())
    return df

if __name__ == "__main__":
    build_golden_set()
