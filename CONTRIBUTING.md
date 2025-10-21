# Contributing to BeaconX-ML

Thank you for your interest in contributing to BeaconX-ML! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Git
- Basic understanding of machine learning and Flask

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/BeaconX-ML.git
   cd BeaconX-ML
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run tests to ensure everything works**
   ```bash
   pytest tests/ -v
   ```

## 📋 Contribution Guidelines

### Code Style
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write comprehensive docstrings
- Keep functions small and focused

### Commit Messages
Use clear, descriptive commit messages:
```
feat: add new earthquake prediction model
fix: resolve coordinate validation bug
docs: update API documentation
test: add unit tests for data processing
```

### Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation if needed

3. **Run tests and linting**
   ```bash
   pytest tests/ -v
   flake8 .
   black --check .
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Provide a clear description of your changes
   - Reference any related issues
   - Include screenshots if applicable

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

### Writing Tests
- Write tests for all new functionality
- Aim for >80% code coverage
- Use descriptive test names
- Test both success and failure cases

## 📝 Documentation

### Code Documentation
- Add docstrings to all functions and classes
- Include type hints
- Explain complex algorithms
- Provide usage examples

### API Documentation
- Update README.md for new endpoints
- Include request/response examples
- Document any breaking changes

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the bug
2. **Steps to reproduce**: Detailed steps to reproduce the issue
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**: OS, Python version, dependencies
6. **Screenshots**: If applicable

## 💡 Feature Requests

When requesting features, please include:

1. **Description**: Clear description of the feature
2. **Use case**: Why this feature would be useful
3. **Implementation ideas**: If you have any ideas
4. **Alternatives**: Any alternative solutions considered

## 🏷️ Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `question`: Further information is requested

## 📞 Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and general discussion
- **Email**: support@beaconx-ml.com

## 🎉 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to BeaconX-ML! 🚀
