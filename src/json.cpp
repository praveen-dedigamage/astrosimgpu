#include "astrosimgpu/json.hpp"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace astrosimgpu {

namespace {
const Json& null_json() {
    static const Json instance;
    return instance;
}
}  // namespace

class JsonParser {
public:
    explicit JsonParser(const std::string& text) : text_(text) {}

    Json parse() {
        skip_ws();
        Json value = parse_value();
        skip_ws();
        if (pos_ != text_.size()) {
            fail("trailing characters after the top-level value");
        }
        return value;
    }

private:
    [[noreturn]] void fail(const std::string& what) const {
        std::ostringstream os;
        os << "JSON parse error at byte " << pos_ << ": " << what;
        throw std::runtime_error(os.str());
    }

    void skip_ws() {
        while (pos_ < text_.size()) {
            const char c = text_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else if (c == '/' && pos_ + 1 < text_.size() && text_[pos_ + 1] == '/') {
                // Line comments are not standard JSON but are handy in a
                // parameter file, so they are tolerated here.
                while (pos_ < text_.size() && text_[pos_] != '\n') {
                    ++pos_;
                }
            } else {
                break;
            }
        }
    }

    char peek() const {
        if (pos_ >= text_.size()) {
            throw std::runtime_error("JSON parse error: unexpected end of input");
        }
        return text_[pos_];
    }

    void expect(char c) {
        if (pos_ >= text_.size() || text_[pos_] != c) {
            fail(std::string("expected '") + c + "'");
        }
        ++pos_;
    }

    Json parse_value() {
        switch (peek()) {
            case '{': return parse_object();
            case '[': return parse_array();
            case '"': return parse_string_value();
            case 't':
            case 'f': return parse_bool();
            case 'n': return parse_null();
            default: return parse_number();
        }
    }

    Json parse_object() {
        Json value;
        value.type_ = Json::Type::Object;
        expect('{');
        skip_ws();
        if (peek() == '}') {
            ++pos_;
            return value;
        }
        while (true) {
            skip_ws();
            const std::string key = parse_string();
            skip_ws();
            expect(':');
            skip_ws();
            value.object_[key] = parse_value();
            skip_ws();
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            expect('}');
            break;
        }
        return value;
    }

    Json parse_array() {
        Json value;
        value.type_ = Json::Type::Array;
        expect('[');
        skip_ws();
        if (peek() == ']') {
            ++pos_;
            return value;
        }
        while (true) {
            skip_ws();
            value.array_.push_back(parse_value());
            skip_ws();
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            expect(']');
            break;
        }
        return value;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            if (pos_ >= text_.size()) {
                fail("unterminated string");
            }
            const char c = text_[pos_++];
            if (c == '"') {
                break;
            }
            if (c != '\\') {
                out.push_back(c);
                continue;
            }
            if (pos_ >= text_.size()) {
                fail("unterminated escape sequence");
            }
            const char esc = text_[pos_++];
            switch (esc) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    if (pos_ + 4 > text_.size()) {
                        fail("truncated \\u escape");
                    }
                    const std::string hex = text_.substr(pos_, 4);
                    pos_ += 4;
                    const unsigned cp =
                        static_cast<unsigned>(std::strtoul(hex.c_str(), nullptr, 16));
                    // Configuration files are ASCII in practice; encode the
                    // basic plane as UTF-8 and leave surrogates alone.
                    if (cp < 0x80) {
                        out.push_back(static_cast<char>(cp));
                    } else if (cp < 0x800) {
                        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                    } else {
                        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                    }
                    break;
                }
                default: fail("unknown escape sequence");
            }
        }
        return out;
    }

    Json parse_string_value() {
        Json value;
        value.type_ = Json::Type::String;
        value.string_ = parse_string();
        return value;
    }

    Json parse_bool() {
        Json value;
        value.type_ = Json::Type::Bool;
        if (text_.compare(pos_, 4, "true") == 0) {
            value.bool_ = true;
            pos_ += 4;
        } else if (text_.compare(pos_, 5, "false") == 0) {
            value.bool_ = false;
            pos_ += 5;
        } else {
            fail("expected true or false");
        }
        return value;
    }

    Json parse_null() {
        if (text_.compare(pos_, 4, "null") != 0) {
            fail("expected null");
        }
        pos_ += 4;
        return Json{};
    }

    Json parse_number() {
        const std::size_t start = pos_;
        if (pos_ < text_.size() && (text_[pos_] == '-' || text_[pos_] == '+')) {
            ++pos_;
        }
        while (pos_ < text_.size()) {
            const char c = text_[pos_];
            if (std::isdigit(static_cast<unsigned char>(c)) || c == '.' || c == 'e' || c == 'E' ||
                c == '+' || c == '-') {
                ++pos_;
            } else {
                break;
            }
        }
        if (pos_ == start) {
            fail("expected a value");
        }
        Json value;
        value.type_ = Json::Type::Number;
        value.number_ = std::strtod(text_.substr(start, pos_ - start).c_str(), nullptr);
        return value;
    }

    const std::string& text_;
    std::size_t pos_ = 0;
};

Json Json::parse(const std::string& text) { return JsonParser(text).parse(); }

Json Json::parse_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open configuration file: " + path);
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return parse(buffer.str());
}

const Json& Json::operator[](const std::string& key) const {
    if (type_ != Type::Object) {
        return null_json();
    }
    const auto it = object_.find(key);
    return it == object_.end() ? null_json() : it->second;
}

bool Json::contains(const std::string& key) const {
    return type_ == Type::Object && object_.find(key) != object_.end();
}

double Json::number(double fallback) const {
    return type_ == Type::Number ? number_ : fallback;
}

bool Json::boolean(bool fallback) const { return type_ == Type::Bool ? bool_ : fallback; }

std::string Json::string(const std::string& fallback) const {
    return type_ == Type::String ? string_ : fallback;
}

void Json::get_to(const std::string& key, double& out) const {
    const Json& v = (*this)[key];
    if (v.type() == Type::Number) {
        out = v.number_;
    }
}

void Json::get_to(const std::string& key, float& out) const {
    const Json& v = (*this)[key];
    if (v.type() == Type::Number) {
        out = static_cast<float>(v.number_);
    }
}

void Json::get_to(const std::string& key, int& out) const {
    const Json& v = (*this)[key];
    if (v.type() == Type::Number) {
        out = static_cast<int>(v.number_);
    }
}

void Json::get_to(const std::string& key, unsigned& out) const {
    const Json& v = (*this)[key];
    if (v.type() == Type::Number) {
        out = static_cast<unsigned>(v.number_);
    }
}

void Json::get_to(const std::string& key, unsigned long& out) const {
    const Json& v = (*this)[key];
    if (v.type() == Type::Number) {
        out = static_cast<unsigned long>(v.number_);
    }
}

void Json::get_to(const std::string& key, unsigned long long& out) const {
    const Json& v = (*this)[key];
    if (v.type() == Type::Number) {
        out = static_cast<unsigned long long>(v.number_);
    }
}

void Json::get_to(const std::string& key, bool& out) const {
    const Json& v = (*this)[key];
    if (v.type() == Type::Bool) {
        out = v.bool_;
    }
}

void Json::get_to(const std::string& key, std::string& out) const {
    const Json& v = (*this)[key];
    if (v.type() == Type::String) {
        out = v.string_;
    }
}

}  // namespace astrosimgpu
