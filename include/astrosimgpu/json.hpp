#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace astrosimgpu {

/// Minimal read-only JSON value.
///
/// Only what the configuration files need: objects, arrays, numbers, strings,
/// booleans and null. Kept in-tree so the simulator builds with nothing but a
/// C++17 compiler, which matters on machines where pulling in a package
/// manager is more trouble than the dependency is worth.
class Json {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Json() = default;

    static Json parse(const std::string& text);
    static Json parse_file(const std::string& path);

    [[nodiscard]] Type type() const { return type_; }
    [[nodiscard]] bool is_null() const { return type_ == Type::Null; }
    [[nodiscard]] bool is_object() const { return type_ == Type::Object; }
    [[nodiscard]] bool is_array() const { return type_ == Type::Array; }

    /// Object lookup. Returns a null Json if the key is absent.
    [[nodiscard]] const Json& operator[](const std::string& key) const;
    [[nodiscard]] bool contains(const std::string& key) const;

    // Keys never asked for. A binary that predates a new option would
    // otherwise ignore it and produce a plausible result, which is worse than
    // failing.
    void collect_unused(const std::string& prefix, std::vector<std::string>& out) const;
    [[nodiscard]] const std::vector<Json>& items() const { return array_; }
    [[nodiscard]] const std::map<std::string, Json>& fields() const { return object_; }

    [[nodiscard]] double number(double fallback) const;
    [[nodiscard]] bool boolean(bool fallback) const;
    [[nodiscard]] std::string string(const std::string& fallback) const;

    /// Assign to `out` only when the key is present, so callers can layer a
    /// file over compiled-in defaults without listing every field.
    void get_to(const std::string& key, double& out) const;
    void get_to(const std::string& key, float& out) const;
    void get_to(const std::string& key, int& out) const;
    void get_to(const std::string& key, unsigned& out) const;
    void get_to(const std::string& key, unsigned long& out) const;
    void get_to(const std::string& key, unsigned long long& out) const;
    void get_to(const std::string& key, bool& out) const;
    void get_to(const std::string& key, std::string& out) const;

private:
    Type type_ = Type::Null;
    mutable std::vector<std::string> asked_;
    bool bool_ = false;
    double number_ = 0.0;
    std::string string_;
    std::vector<Json> array_;
    std::map<std::string, Json> object_;

    friend class JsonParser;
};

}  // namespace astrosimgpu
