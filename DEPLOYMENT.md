# Deployment Guide

## GitHub Actions Deployment

The personal-chatter application uses GitHub Actions for automated CI/CD. 

### Workflows

1. **CI/CD Workflow** (`.github/workflows/ci-cd.yml`)
   - Runs on Python 3.12
   - Tests on Ubuntu and Windows
   - Runs linting, testing, and code analysis

2. **Deploy Workflow** (`.github/workflows/deploy.yml`)
   - Builds Docker image
   - Pushes to GitHub Container Registry
   - Deploys to production (when pushed to main branch)

### Required GitHub Secrets

For production deployment, set these secrets in your GitHub repository:

```
DATABASE_URL=postgresql://user:password@host:port/database
REDIS_URL=redis://host:port/database
SECRET_KEY=your_secure_secret_key_here
JWT_SECRET=your_jwt_secret_here
BRAVE_SEARCH_API_KEY=your_brave_search_api_key
```

### Environment Files

- `.env.development` - Development configuration
- `.env.production` - Production configuration template

### Docker Deployment

#### Using Docker Compose

1. **Development:**
   ```bash
   docker-compose --profile development up
   ```

2. **Production:**
   ```bash
   docker-compose --profile production up -d
   ```

3. **With monitoring:**
   ```bash
   docker-compose --profile production --profile monitoring up -d
   ```

#### Manual Docker Build

```bash
# Build the image
docker build -t personal-chatter .

# Run the container
docker run -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e DATABASE_URL="your_database_url" \
  -e REDIS_URL="your_redis_url" \
  personal-chatter
```

### Health Checks

The application exposes a health check endpoint at `/health` for monitoring.

### Monitoring

- Prometheus metrics on port 9090
- Grafana dashboard on port 3000
- Application metrics on port 8001

### Troubleshooting

1. **Build Failures:**
   - Check Python version compatibility (requires 3.12)
   - Verify all dependencies in requirements.txt
   - Check Dockerfile syntax

2. **Runtime Issues:**
   - Verify environment variables are set
   - Check database connectivity
   - Review application logs

3. **GitHub Actions Issues:**
   - Check repository secrets are configured
   - Verify workflow permissions
   - Review action logs for specific errors
