# Security Policy

## 📦 Version Support

We only support the latest release. Please ensure you're using the most recent version of Personal Chatter to receive security updates and support.

## 🚨 Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please follow these steps:

### 📧 Private Disclosure

**Please do NOT create public GitHub issues for security vulnerabilities.**

Instead, either:
- Fix the issue yourself and submit a pull request
- Message me on Discord: **@ignemia**

If you choose to report the issue, include the following information:
- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Suggested mitigation (if any)
- Your contact information

### 🕒 Response Timeline

- **24-48 hours**: Initial acknowledgment of your report
- **1-2 weeks**: Investigation and fix development (depending on severity)
- **2-4 weeks**: Security patch release (for high/critical issues)

### 🎯 Scope

Security issues we're particularly interested in:

#### ✅ In Scope
- Remote code execution vulnerabilities
- Authentication and authorization bypasses
- Data exposure or privacy violations
- Cross-site scripting (XSS) in web interfaces
- Insecure cryptographic implementations
- Dependency vulnerabilities with exploitable impact
- Model poisoning or adversarial attacks
- Prompt injection attacks

#### ❌ Out of Scope
- Social engineering attacks
- Physical security issues
- Denial of service (DoS) attacks
- Issues in third-party dependencies (report to maintainers)
- Theoretical vulnerabilities without practical exploitation
- Issues requiring physical access to the machine
- **Your machine's security** - secure your own system
- **Your network security** - that's your responsibility
- **Your personal operational security** - take responsibility for this

**Note:** We only handle security issues that can be fixed in our code. If your machine, home network, or other infrastructure is insecure, that's your responsibility to address.

### 🏆 Recognition

We believe in recognizing security researchers who help make Personal Chatter safer:

- **Credits**: Mention in release notes and CONTRIBUTORS.md

## 🛡️ Security Best Practices

### For Users
- Keep Personal Chatter updated to the latest version
- Use strong, unique passwords for any authentication
- **Secure your own system and network** - this is your responsibility
- Regularly review and audit your data and conversations
- Monitor system logs for unusual activity
- Be cautious with custom models and third-party integrations

### For Developers
- Follow secure coding practices
- Validate all inputs and sanitize outputs
- Use parameterized queries for database operations
- Implement proper authentication and authorization
- Keep dependencies updated and scan for vulnerabilities
- Use secure communication protocols (HTTPS, WSS)

## 📚 Security Resources

- **OWASP Top 10**: https://owasp.org/www-project-top-ten/
- **NIST Cybersecurity Framework**: https://www.nist.gov/cyberframework

## 📞 Contact Information

- **Security Issues**: Message on Discord @ignemia or submit a pull request
- **General Contact**: Message on Discord @ignemia

---

*This security policy is subject to change. Last updated: January 2025*
