"""
Synthetic Data Generator for Customer Support on Twitter (twcs.csv fallback).

Generates a realistic ~2,400-conversation (4,800-row) dataset following the exact
schema of Kaggle's `thoughtvector/customer-support-on-twitter` (twcs.csv).
Used when the raw Kaggle dataset is not locally downloaded, ensuring zero-setup reproducibility.
"""

import os
import random
import pandas as pd
from datetime import datetime, timedelta

SCHEMA_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id"
]

BRAND = "AmazonHelp"

# Real-world templates across 8 intent archetypes
INTENT_TEMPLATES = {
    "order_delivery_delay": {
        "customer": [
            "@{brand} Where is my package? Order #{order_id} was supposed to arrive {time_phrase}, still nothing!",
            "@{brand} My tracking says delivered but I have not received anything for order #{order_id}!",
            "@{brand} shipment #{order_id} is delayed by 3 days already. Any update on when it will arrive?",
            "@{brand} Carrier marked order #{order_id} as delivered at front door but there is no package anywhere.",
            "@{brand} Package #{order_id} has been stuck in transit in transit hub for over 48 hours. What gives?",
            "@{brand} Ordered express next-day delivery on #{order_id} and it still hasn't shipped!",
            "@{brand} Tracking status says delivery attempted but nobody buzzed or knocked. Order #{order_id}.",
            "@{brand} Can someone check on order #{order_id}? It's a birthday gift and now it's late.",
        ],
        "brand": [
            "Hi there, we're sorry to hear your order #{order_id} is delayed! Please send us a DM at amzn.to/help with your details so we can investigate.",
            "Hello! We apologize for the delivery issue with #{order_id}. Please check around your porch/neighbors, and if still missing, DM us your account email.",
            "We understand your concern regarding order #{order_id}. Please drop us a DM with your tracking link and postal code so we can check with the carrier.",
            "So sorry for the delay on #{order_id}! Please DM us your order details and we will check the transit status directly for you.",
        ]
    },
    "refund_request": {
        "customer": [
            "@{brand} I returned item on order #{order_id} over a week ago and still haven't received my refund.",
            "@{brand} Please cancel my order #{order_id} and process a full refund immediately.",
            "@{brand} I was told my refund of ${amount} for order #{order_id} would be in my account in 3-5 days. It's been 10 days.",
            "@{brand} Received broken item in order #{order_id}. I want my money back, not a replacement.",
            "@{brand} How long does it take for a refund to process to my original payment method for return #{order_id}?",
            "@{brand} Return status says received at warehouse but refund not initiated for #{order_id}. Please help.",
            "@{brand} Charged for an item I sent back last Monday. When will my refund of ${amount} arrive?",
            "@{brand} Need a refund for #{order_id}. Sent back via drop-off locker 5 days ago.",
        ],
        "brand": [
            "Hello! Once an item is received at our fulfillment center, refunds usually take 3-5 business days. Please DM us your order #{order_id} to check status.",
            "We're sorry for the delay in your refund! Please reach out to us in DM with order #{order_id} and your email address so we can review the return.",
            "Hi! We want to help get this sorted. Please send us a direct message with order #{order_id} and return tracking number so we can verify with our billing team.",
            "Apologies for the inconvenience! Please DM us your account details so we can check on return #{order_id} and expedite your refund.",
        ]
    },
    "account_access": {
        "customer": [
            "@{brand} Locked out of my account. The OTP verification code is not being sent to my phone.",
            "@{brand} I forgot my password and the password reset link is never arriving in my inbox.",
            "@{brand} My account was compromised or hacked, someone changed the email on file. Need help urgently!",
            "@{brand} Two-factor authentication is failing, says invalid token even though it's correct.",
            "@{brand} Unable to sign in to my account from my new laptop. Says suspicious activity detected.",
            "@{brand} Lost access to my old phone number and now 2FA won't let me log into my account.",
            "@{brand} Every time I try to log in, it loops back to the login screen without an error message.",
            "@{brand} Help! Account locked due to too many failed attempts, how do I unlock it?",
        ],
        "brand": [
            "Hi! For your security, please don't share passwords publicly. Please visit amzn.to/account-recovery or DM us to begin identity verification.",
            "We're sorry you're having trouble accessing your account. Please send us a DM so we can securely guide you through the two-step verification reset.",
            "Hello! If you suspect unauthorized access, please visit amzn.to/security immediately. You can also DM us your associated phone or email.",
            "Hi there, please ensure SMS isn't blocked by your carrier, or try clearing browser cache. If the issue persists, send us a DM.",
        ]
    },
    "billing_issue": {
        "customer": [
            "@{brand} I was charged ${amount} twice for order #{order_id}. Please check your billing system.",
            "@{brand} Just noticed an unauthorized charge of ${amount} for Prime membership that I never authorized.",
            "@{brand} Why was my credit card charged ${amount} when my order total was only ${amount_small}?",
            "@{brand} My subscription auto-renewed even though I turned off auto-renew last week!",
            "@{brand} Can you explain the invoice breakdown for #{order_id}? There's an extra charge of ${amount}.",
            "@{brand} Payment declined on my card even though there are sufficient funds available.",
            "@{brand} Charged ${amount} annual subscription fee by mistake, please reverse this charge.",
            "@{brand} You guys billed me twice on the same day for order #{order_id}. Please fix this duplicate charge.",
        ],
        "brand": [
            "Hi! We'd be glad to look into this charge for you. Please send us a DM with the date and amount so we can locate the transaction securely.",
            "Hello, duplicate charges are often pending authorizations that drop off in 2-3 business days. Please DM us your order #{order_id} to confirm.",
            "We apologize for the billing confusion. Please reach out via DM with the last 4 digits of the card and transaction date so we can inspect.",
            "Sorry for the unexpected charge! If you did not intend to renew Prime, please DM us your email and we can help reverse eligible fees.",
        ]
    },
    "technical_bug": {
        "customer": [
            "@{brand} Your iOS app keeps crashing every time I tap the checkout button. Version 18.2.",
            "@{brand} Website gives error 500 internal server error when trying to view my order history.",
            "@{brand} Search bar on the website is completely unresponsive on Chrome desktop.",
            "@{brand} The mobile app is not loading product images, just showing blank white boxes.",
            "@{brand} Getting 'An unexpected error occurred' whenever I try to add items to my shopping cart.",
            "@{brand} The video streaming player on Prime Video keeps stuttering and buffering on my Smart TV.",
            "@{brand} Wishlist won't let me remove deleted items, says 'Failed to update wishlist'.",
            "@{brand} Checkout page freezes when selecting saved shipping address.",
        ],
        "brand": [
            "Hi! Thanks for bringing this to our attention. Have you tried reinstalling the app or clearing your cache? If so, please DM us your device OS version.",
            "We apologize for the technical glitch! Our technical team is monitoring site stability. Please send us a DM with a screenshot if the error continues.",
            "Hello! Please try accessing the site in an incognito window or restarting your device. If the bug persists, DM us your app version and device model.",
            "We're sorry for the app crash. Please ensure you are running the latest app update. DM us if you need further technical troubleshooting.",
        ]
    },
    "general_inquiry": {
        "customer": [
            "@{brand} Do you ship internationally to Germany and what are the customs fees?",
            "@{brand} Is the manufacturer warranty valid if I buy an electronic item through third party sellers?",
            "@{brand} Can I use Amazon gift cards to purchase digital books on Kindle?",
            "@{brand} What is the return window for holiday purchases made in November?",
            "@{brand} Do you offer student discounts on annual Prime memberships?",
            "@{brand} How can I change my primary delivery address before placing an order?",
            "@{brand} Are recyclable packaging options available for fragile items?",
            "@{brand} Does this product come with a power adapter or do I need to buy it separately?",
        ],
        "brand": [
            "Hi! You can check international shipping options and estimated import duties at checkout or by visiting amzn.to/shipping-rates.",
            "Hello! Most electronics purchased from authorized sellers include standard warranty. Please DM us the product ASIN to check details.",
            "Hi there! Yes, Amazon gift cards can be applied to Kindle digital purchases. Check out amzn.to/giftcard-faq for more information.",
            "Hello! Our standard return policy is 30 days, but extended holiday return windows often apply. Let us know if you have a specific item in mind!",
        ]
    },
    "complaint_escalation": {
        "customer": [
            "@{brand} WORST CUSTOMER SERVICE EVER! Representative hung up on me after waiting 45 minutes on hold!",
            "@{brand} This is completely unacceptable. I want to speak to a supervisor or corporate manager right now!",
            "@{brand} You have lied to me three times about my delivery. I am reporting this fraud to the Better Business Bureau.",
            "@{brand} If my issue is not resolved today, I will be taking legal action with my attorney. Order #{order_id}.",
            "@{brand} Your support agent was extremely rude and unhelpful. Disgraceful treatment of a 10-year customer.",
            "@{brand} Absolute scam. Stole my money and refusing to help. Give me someone who actually has authority.",
            "@{brand} I demand an immediate escalation to executive customer relations. No more generic bot answers!",
            "@{brand} Unbelievable negligence. Damaged goods, ruined anniversary, zero accountability from your team.",
        ],
        "brand": [
            "We sincerely apologize for this distressing experience. This is certainly not the level of service we aim to provide. Please DM us immediately so a senior specialist can assist.",
            "We are very sorry to hear this and take feedback like this seriously. Please send us a direct message with your account details and phone number so we can escalate to a supervisor.",
            "We hear your frustration and apologize for the breakdown in communication. Please DM us your order #{order_id} so we can urgently review this with our leadership team.",
            "This is definitely not the experience we want for our customers. Please reach out in DM with your contact information so an escalation manager can connect with you.",
        ]
    },
    "positive_feedback": {
        "customer": [
            "@{brand} Huge shoutout to Sarah from customer support who solved my lost package in 5 minutes! Amazing service!",
            "@{brand} Thank you so much for the quick replacement on order #{order_id}. You guys are the best!",
            "@{brand} Incredible customer support experience today. Truly appreciate how fast my issue was handled.",
            "@{brand} Just wanted to say thank you for delivering my package ahead of schedule! Great job!",
            "@{brand} Your support team went above and beyond today. Thank you for the courteous and prompt help!",
            "@{brand} Amazon customer service never disappoints. Replaced my broken item with no hassle. Thanks!",
            "@{brand} 5 stars to the agent who helped me reset my account this morning. Great experience!",
            "@{brand} Thank you @{brand} for resolving my billing query so smoothly today!",
        ],
        "brand": [
            "Thank you so much for the kind words! We love hearing this and will pass your compliments along to the team. Have a wonderful day!",
            "We're thrilled to hear Sarah was able to help! Thank you for choosing us, and please don't hesitate to reach out if you ever need anything else.",
            "You're very welcome! We're so glad everything was resolved smoothly. Thank you for being a valued customer!",
            "Hearing this makes our day! Thank you for your feedback and support. We're always here to help!",
        ]
    }
}

def generate_synthetic_dataset(output_path: str, n_conversations: int = 2400, seed: int = 42) -> pd.DataFrame:
    """
    Generates realistic customer support tweet conversations paired with brand replies.
    """
    random.seed(seed)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    rows = []
    start_date = datetime(2017, 10, 1, 9, 0, 0)
    current_tweet_id = 100000

    intents = list(INTENT_TEMPLATES.keys())
    # Slightly unequal distribution to mirror realistic customer care volume
    weights = [0.26, 0.18, 0.12, 0.14, 0.10, 0.08, 0.07, 0.05]

    for i in range(n_conversations):
        intent = random.choices(intents, weights=weights)[0]
        customer_templates = INTENT_TEMPLATES[intent]["customer"]
        brand_templates = INTENT_TEMPLATES[intent]["brand"]

        # Randomize slot fillers
        order_id = f"{random.randint(100, 999)}-{random.randint(1000000, 9999999)}-{random.randint(1000000, 9999999)}"
        amount = f"{random.randint(15, 250)}.{random.choice(['00', '49', '99', '50'])}"
        amount_small = f"{random.randint(5, 30)}.{random.choice(['00', '99'])}"
        time_phrase = random.choice(["yesterday", "2 days ago", "this morning", "on Friday", "last week"])

        cust_text_template = random.choice(customer_templates)
        cust_text = cust_text_template.format(
            brand=BRAND,
            order_id=order_id,
            amount=amount,
            amount_small=amount_small,
            time_phrase=time_phrase
        )

        brand_text_template = random.choice(brand_templates)
        brand_text = brand_text_template.format(
            order_id=order_id,
            amount=amount
        )

        cust_tweet_id = current_tweet_id
        brand_tweet_id = current_tweet_id + 1
        current_tweet_id += 2

        author_id = f"cust_{random.randint(100000, 999999)}"
        cust_time = start_date + timedelta(minutes=random.randint(1, 40000))
        brand_time = cust_time + timedelta(minutes=random.randint(3, 45))

        time_format = "%a %b %d %H:%M:%S +0000 %Y"

        # Inbound tweet (Customer)
        rows.append({
            "tweet_id": cust_tweet_id,
            "author_id": author_id,
            "inbound": True,
            "created_at": cust_time.strftime(time_format),
            "text": cust_text,
            "response_tweet_id": brand_tweet_id,
            "in_response_to_tweet_id": None
        })

        # Outbound tweet (Brand reply)
        rows.append({
            "tweet_id": brand_tweet_id,
            "author_id": BRAND,
            "inbound": False,
            "created_at": brand_time.strftime(time_format),
            "text": brand_text,
            "response_tweet_id": None,
            "in_response_to_tweet_id": cust_tweet_id
        })

    df = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    df.to_csv(output_path, index=False)
    print(f"[make_sample_data] Generated {len(df)} rows ({n_conversations} conversation pairs) to {output_path}")
    return df

if __name__ == "__main__":
    target = os.path.join("data", "raw", "twcs.csv")
    generate_synthetic_dataset(target)
