---
layout: tag
title: "Tag: Engineering"
permalink: /t/engineering
---

<ul class="post-list">
  {%- for post in site.tags["Engineering"] -%}
    <li>
      {%- assign date_format = site.minima.date_format | default: "%Y-%m-%d" -%}
      <span class="post-meta">
        {{ post.date | date: date_format }}
      </span>
      <a class="post-link" href="{{ post.url | relative_url }}">
          {{ post.title | escape }}
      </a>
    </li>
  {%- endfor -%}
</ul>