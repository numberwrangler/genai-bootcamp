# Custom Certificate Configuration

If you want to make your digital twin look more professional, you can use a custom domain.
To do this, you need a registered domain which you can update the records for. It does not have to be
registered or hosted via Route53 (the AWS DNS provider), but the process is slightly easier if it is.
You will need an SSL certificate for your custom domain from AWS, which is free.

1. Log in to the AWS console and switch to the `us-east-1` region.
2. Open the ACM console at https://console.aws.amazon.com/acm/home and choose Request a certificate.
3. Select `Request a public certificate` and click Next.
4. Under `Fully qualified domain name` enter either the full domain name you will be using (eg `digitaltwin.example.com`) or a wildcard (`*.example.com`). A wildcard is preferable.
5. Ensure 'Disable export' is selected to ensure only a free certificate is created
6. Leave `DNS validation` selected and use the default key algorithm and click `Request`.
7. On the next screen, you'll see your certificate requests. Click on the domain name to view its details.
8. In the certificate details page, you'll see a "Domains" section with a "Validation" column. Click the "Create records in Route 53" button if your domain is hosted in Route 53. If not, click "Copy to clipboard" to get the DNS validation records.
9. If you're using Route 53:
   - Click "Create records in Route 53"
   - Confirm the creation of the CNAME records
   - The records will be automatically added to your hosted zone
10. If you're not using Route 53:
    - Copy the DNS validation records (CNAME records) 
    - Add these records to your DNS provider's configuration
    - The record name and value will be provided in the AWS console
11. Wait for DNS propagation (this can take a few minutes to several hours)
12. The certificate status will change from "Pending validation" to "Issued" once validation is complete

---

# Custom domain configuration

## Update the CloudFront configuration

In [app.py](week8/1-admin-ui/app.py) add two parameters to the `Twin` stack:

- `custom_domain_name`: This should match the full domain name you want to host your twin at (eg dtwin.example.com)
- `custom_certificate_arn`: This should be the [ARN](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference-arns.html) for the certificate you generated above

For example:

```python
twin_stack = Twin(
    app,
    "Twin",
    # ... existing parameters ...
    custom_domain_name="dtwin.example.com",  # Replace with your domain
    custom_certificate_arn="arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012"  # Replace with your certificate ARN
)
```

Run cdk (via Github Actions, or via `uv run cdk deploy --all`).

## Create a CNAME record

Add a [CNAME record](https://en.wikipedia.org/wiki/CNAME_record) from your full domain to your CloudFront distribution name.
For example, if your CloudFront distribution is at `d2wmcljp6gwf0d.cloudfront.net` and you are making your twin available
at `twin.example.com`, add a CNAME record as follows:

```
twin CNAME d2wmcljp6gwf0d.cloudfront.net.
```

Depending on your DNS provider, the trailing `.` may or may not be allowed or required.
For TTL any value is OK - most providers will default to 300 seconds (5 minutes).
